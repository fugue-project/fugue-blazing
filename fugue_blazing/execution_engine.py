import logging
from threading import RLock
from typing import Any, Callable, List, Optional, Set, Union

import cudf
from blazingsql import BlazingContext
from triad.collections.dict import IndexedOrderedDict
from triad.collections.fs import FileSystem
from triad.utils.assertion import assert_or_throw

from fugue import NativeExecutionEngine
from fugue.collections.partition import (
    EMPTY_PARTITION_SPEC,
    PartitionCursor,
    PartitionSpec,
    parse_presort_exp,
)
from fugue.dataframe import (
    DataFrame,
    DataFrames,
    LocalBoundedDataFrame,
    LocalDataFrame,
    PandasDataFrame,
)
from fugue.dataframe.utils import get_join_schemas
from fugue.exceptions import FugueError
from fugue.execution.execution_engine import (
    _DEFAULT_JOIN_KEYS,
    ExecutionEngine,
    SQLEngine,
)
from fugue_blazing._io import load_df, save_df
from fugue_blazing.dataframe import CudaDataFrame


class BlazingSQLEngine(SQLEngine):
    """BlazingSQL engine.

    :param execution_engine: the execution engine this sql engine will run on
    """

    def __init__(self, execution_engine: ExecutionEngine):
        super().__init__(execution_engine)

    def select(self, dfs: DataFrames, statement: str) -> DataFrame:
        result = self.execution_engine.blazing_sql(statement, **dfs)
        return self.execution_engine.to_df(result)


class CudaExecutionEngine(ExecutionEngine):
    """The execution engine using local GPU.
    Please read |ExecutionEngineTutorial| to understand this important Fugue concept
    :param conf: |ParamsLikeObject|, read |FugueConfig| to learn Fugue specific options
    """

    def __init__(self, conf: Any = None, context: Optional[BlazingContext] = None):
        super().__init__(conf)
        self._native_engine = NativeExecutionEngine(conf)
        self._context = context or BlazingContext()
        self._context_lock = RLock()

    def __repr__(self) -> str:
        return "CudaExecutionEngine"

    @property
    def context(self) -> BlazingContext:
        return self._context

    def blazing_sql(self, sql: str, **dfs: DataFrame) -> cudf.DataFrame:
        tables: Set[str] = set()
        with self._context_lock:
            try:
                for k, v in dfs.items():
                    if isinstance(v, (PandasDataFrame, CudaDataFrame)):
                        self.context.create_table(k, v.native)
                    else:
                        self.context.create_table(k, self.to_df(v).native)
                    tables.add(k)
                df = self.context.sql(sql)
            finally:
                for table in tables:
                    self.context.drop_table(table)
        assert_or_throw(
            isinstance(df, cudf.DataFrame),
            FugueError(f"Blazing SQL can't handle {sql}"),
        )
        return df.reset_index(drop=True)

    @property
    def log(self) -> logging.Logger:
        return self._native_engine.log

    @property
    def fs(self) -> FileSystem:
        return self._native_engine.fs

    @property
    def default_sql_engine(self) -> SQLEngine:
        return BlazingSQLEngine(self)

    def to_df(self, df: Any, schema: Any = None, metadata: Any = None) -> CudaDataFrame:
        if isinstance(df, DataFrame):
            assert_or_throw(
                schema is None and metadata is None,
                ValueError("schema and metadata must be None when df is a DataFrame"),
            )
            if isinstance(df, CudaDataFrame):
                return df
            if isinstance(df, PandasDataFrame):
                return CudaDataFrame(df.native, df.schema, df.metadata)
            return CudaDataFrame(df.as_pandas(), df.schema, df.metadata)
        return CudaDataFrame(df, schema, metadata)

    def repartition(
        self, df: DataFrame, partition_spec: PartitionSpec
    ) -> DataFrame:  # pragma: no cover
        self.log.warning("%s doesn't respect repartition", self)
        return df

    def map(
        self,
        df: DataFrame,
        map_func: Callable[[PartitionCursor, LocalDataFrame], LocalDataFrame],
        output_schema: Any,
        partition_spec: PartitionSpec,
        metadata: Any = None,
        on_init: Optional[Callable[[int, DataFrame], Any]] = None,
    ) -> DataFrame:
        return self._native_engine.map(
            df=df,
            map_func=map_func,
            output_schema=output_schema,
            partition_spec=partition_spec,
            metadata=metadata,
            on_init=on_init,
        )

    def broadcast(self, df: DataFrame) -> DataFrame:
        return self.to_df(df)

    def persist(
        self,
        df: DataFrame,
        lazy: bool = False,
        **kwargs: Any,
    ) -> DataFrame:
        return self.to_df(df)

    def join(
        self,
        df1: DataFrame,
        df2: DataFrame,
        how: str,
        on: List[str] = _DEFAULT_JOIN_KEYS,
        metadata: Any = None,
    ) -> DataFrame:
        key_schema, output_schema = get_join_schemas(df1, df2, how=how, on=on)
        on_fields = " AND ".join(f"df1.{k}=df2.{k}" for k in key_schema)
        join_type = self._how_to_join(how)
        if how.lower() == "cross":
            select_fields = ",".join(
                f"df1.{k}" if k in df1.schema else f"df2.{k}"
                for k in output_schema.names
            )
            sql = f"SELECT {select_fields} FROM df1 {join_type} df2"
        elif how.lower() == "right_outer":
            select_fields = ",".join(
                f"df2.{k}" if k in df2.schema else f"df1.{k}"
                for k in output_schema.names
            )
            sql = f"SELECT {select_fields} FROM df2 LEFT OUTER JOIN df1 ON {on_fields}"
        elif how.lower() == "full_outer":
            select_fields = ",".join(
                f"COALESCE(df1.{k},df2.{k}) AS {k}" if k in key_schema else k
                for k in output_schema.names
            )
            sql = f"SELECT {select_fields} FROM df1 {join_type} df2 ON {on_fields}"
        elif how.lower() in ["semi", "left_semi"]:
            keys = ",".join(key_schema.names)
            on_fields = " AND ".join(f"df1.{k}=df3.{k}" for k in key_schema)
            sql = (
                f"SELECT df1.* FROM df1 INNER JOIN (SELECT DISTINCT {keys} "
                f"FROM df2) AS df3 ON {on_fields}"
            )
        elif how.lower() in ["anti", "left_anti"]:
            keys = ",".join(key_schema.names)
            on_fields = " AND ".join(f"df1.{k}=df3.{k}" for k in key_schema)
            sql = (
                "SELECT df1.* FROM df1 LEFT OUTER JOIN "
                f"(SELECT DISTINCT {keys}, 1 AS __contain__ FROM df2) AS df3 "
                f"ON {on_fields} WHERE df3.__contain__ IS NULL"
            )
        else:
            select_fields = ",".join(
                f"df1.{k}" if k in df1.schema else f"df2.{k}"
                for k in output_schema.names
            )
            sql = f"SELECT {select_fields} FROM df1 {join_type} df2 ON {on_fields}"
        result = self.blazing_sql(sql, df1=df1, df2=df2)
        return CudaDataFrame(result, output_schema, metadata)

    def _how_to_join(self, how: str):
        return how.upper().replace("_", " ") + " JOIN"

    def union(
        self,
        df1: DataFrame,
        df2: DataFrame,
        distinct: bool = True,
        metadata: Any = None,
    ) -> DataFrame:
        assert_or_throw(
            df1.schema == df2.schema, ValueError(f"{df1.schema} != {df2.schema}")
        )
        union_type = "UNION DISTINCT" if distinct else "UNION ALL"
        sql = f"SELECT * FROM df1 {union_type} SELECT * FROM df2"
        result = self.blazing_sql(sql, df1=df1, df2=df2)
        return CudaDataFrame(result, df1.schema, metadata=metadata)

    def subtract(
        self,
        df1: DataFrame,
        df2: DataFrame,
        distinct: bool = True,
        metadata: Any = None,
    ) -> DataFrame:  # pragma: no cover
        raise NotImplementedError("Blazing SQL doesn't support EXCEPT")

    def intersect(
        self,
        df1: DataFrame,
        df2: DataFrame,
        distinct: bool = True,
        metadata: Any = None,
    ) -> DataFrame:  # pragma: no cover
        raise NotImplementedError("Blazing SQL doesn't support INTERSECT")

    def distinct(
        self,
        df: DataFrame,
        metadata: Any = None,
    ) -> DataFrame:
        sql = "SELECT DISTINCT * FROM df"
        result = self.blazing_sql(sql, df=df)
        return CudaDataFrame(result, df.schema, metadata=metadata)

    def dropna(
        self,
        df: DataFrame,
        how: str = "any",
        thresh: int = None,
        subset: List[str] = None,
        metadata: Any = None,
    ) -> DataFrame:
        d = self.to_df(df).native.dropna(
            axis=0, how=how, thresh=thresh, subset=subset, inplace=False
        )
        return CudaDataFrame(d.reset_index(drop=True), df.schema, metadata)

    def fillna(
        self,
        df: DataFrame,
        value: Any,
        subset: List[str] = None,
        metadata: Any = None,
    ) -> DataFrame:
        assert_or_throw(
            (not isinstance(value, list)) and (value is not None),
            ValueError("fillna value can not None or a list"),
        )
        if isinstance(value, dict):
            assert_or_throw(
                (None not in value.values()) and (any(value.values())),
                ValueError(
                    "fillna dict can not contain None and needs at least one value"
                ),
            )
            mapping = value
        else:
            # If subset is none, apply to all columns
            subset = subset or df.schema.names
            mapping = {col: value for col in subset}
        d = self.to_df(df).native.fillna(mapping, inplace=False)
        return CudaDataFrame(d.reset_index(drop=True), df.schema, metadata)

    def sample(
        self,
        df: DataFrame,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        seed: Optional[int] = None,
        metadata: Any = None,
    ) -> DataFrame:
        assert_or_throw(
            (n is None and frac is not None) or (n is not None and frac is None),
            ValueError("one and only one of n and frac should be set"),
        )
        d = self.to_df(df).native.sample(
            n=n, frac=frac, replace=replace, random_state=seed
        )
        return CudaDataFrame(d.reset_index(drop=True), df.schema, metadata)

    def take(
        self,
        df: DataFrame,
        n: int,
        presort: str,
        na_position: str = "last",
        partition_spec: PartitionSpec = EMPTY_PARTITION_SPEC,
        metadata: Any = None,
    ) -> DataFrame:
        assert_or_throw(
            isinstance(n, int),
            ValueError("n needs to be an integer"),
        )
        d = self.to_df(df).native

        # Use presort over partition_spec.presort if possible
        if presort:
            presort = parse_presort_exp(presort)
        _presort: IndexedOrderedDict = presort or partition_spec.presort

        if len(_presort.keys()) > 0:
            # cudf doesn't support an array of ascending
            # here is a treatment for all True or all False
            if all(x for x in _presort.values()):
                asc: Any = True
            elif all(not x for x in _presort.values()):
                asc = False
            else:
                # and we don't raise here because
                # it's unknown when cudf will start supporting it
                asc = list(_presort.values())
            d = d.sort_values(
                list(_presort.keys()),
                ascending=asc,
                na_position=na_position,
            )

        if len(partition_spec.partition_by) == 0:
            d = d.head(n)
        else:
            d = d.groupby(by=partition_spec.partition_by, dropna=False).apply(
                lambda sub: sub.head(n)
            )

        return CudaDataFrame(d.reset_index(drop=True), df.schema, metadata)

    def load_df(
        self,
        path: Union[str, List[str]],
        format_hint: Any = None,
        columns: Any = None,
        **kwargs: Any,
    ) -> LocalBoundedDataFrame:
        return self.to_df(
            load_df(
                path, format_hint=format_hint, columns=columns, fs=self.fs, **kwargs
            )
        )

    def save_df(
        self,
        df: DataFrame,
        path: str,
        format_hint: Any = None,
        mode: str = "overwrite",
        partition_spec: PartitionSpec = EMPTY_PARTITION_SPEC,
        force_single: bool = False,
        **kwargs: Any,
    ) -> None:
        if not partition_spec.empty:
            self.log.warning(  # pragma: no cover
                "partition_spec is not respected in %s.save_df", self
            )
        df = self.to_df(df).as_local()
        save_df(df, path, format_hint=format_hint, mode=mode, fs=self.fs, **kwargs)
