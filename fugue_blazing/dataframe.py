from typing import Any, Dict, Iterable, List, Optional, Tuple

import cudf
import pandas
import pyarrow as pa
from fugue_blazing._utils import CUDF_UTILS
from triad import assert_or_throw
from triad.collections.schema import Schema
from triad.utils.pandas_like import PD_UTILS

from fugue import DataFrame, LocalDataFrame, PandasDataFrame
from fugue.dataframe.dataframe import _input_schema
from fugue.exceptions import FugueDataFrameInitError, FugueDataFrameOperationError


class CudaDataFrame(DataFrame):
    """DataFrame that wraps CUDF. Please also read
    |DataFrameTutorial| to understand this Fugue concept
    :param df: :class:`dask:dask.dataframe.DataFrame`,
      pandas DataFrame or list or iterable of arrays
    :param schema: |SchemaLikeObject| or :class:`spark:pyspark.sql.types.StructType`,
      defaults to None.
    :param metadata: |ParamsLikeObject|, defaults to None
    :param num_partitions: initial number of partitions for the dask dataframe
      defaults to 0 to get the value from `fugue.dask.dataframe.default.partitions`
    :param type_safe: whether to cast input data to ensure type safe, defaults to True
    :raises FugueDataFrameInitError: if the input is not compatible
    :Notice:
    * For :class:`dask:dask.dataframe.DataFrame`, schema must be None
    """

    def __init__(  # noqa: C901
        self,
        df: Any = None,
        schema: Any = None,
        metadata: Any = None,
        type_safe: bool = True,
    ):
        try:
            if df is None:
                schema = _input_schema(schema).assert_not_empty()
                df = []
            if isinstance(df, CudaDataFrame):
                super().__init__(
                    df.schema, df.metadata if metadata is None else metadata
                )
                self._native: cudf.DataFrame = df._native
                return
            elif isinstance(df, (cudf.DataFrame, cudf.Series)):
                if isinstance(df, cudf.Series):
                    df = df.to_frame()
                pdf = df
                schema = None if schema is None else _input_schema(schema)
            elif isinstance(df, (pandas.DataFrame, pandas.Series)):
                if isinstance(df, pandas.Series):
                    df = df.to_frame()
                pdf = cudf.from_pandas(df)
                schema = None if schema is None else _input_schema(schema)
            elif isinstance(df, Iterable):
                schema = _input_schema(schema).assert_not_empty()
                t = PandasDataFrame(df, schema)
                pdf = cudf.from_pandas(t.native)
            else:
                raise ValueError(f"{df} is incompatible with DaskDataFrame")
            pdf, schema = self._apply_schema(pdf, schema)
            super().__init__(schema, metadata)
            self._native = pdf
        except Exception as e:
            raise FugueDataFrameInitError from e

    @property
    def native(self) -> cudf.DataFrame:
        """The wrapped cuda dataframe
        :rtype: :class:`cudf:cudf.core.dataframe.DataFrame`
        """
        return self._native

    @property
    def is_local(self) -> bool:
        return False

    def as_local(self) -> LocalDataFrame:
        return PandasDataFrame(self.as_pandas(), self.schema, self.metadata)

    @property
    def is_bounded(self) -> bool:
        return True

    @property
    def num_partitions(self) -> int:  # pragma: no cover
        return 1

    @property
    def empty(self) -> bool:
        return CUDF_UTILS.empty(self.native)

    def _drop_cols(self, cols: List[str]) -> DataFrame:
        cols = (self.schema - cols).names
        return self._select_cols(cols)

    def _select_cols(self, cols: List[Any]) -> DataFrame:
        schema = self.schema.extract(cols)
        return CudaDataFrame(self.native[schema.names], schema)

    def peek_array(self) -> Any:
        self.assert_not_empty()
        return self._native.iloc[:1].to_pandas().iloc[0].values.tolist()

    def count(self) -> int:
        return self._native.shape[0]

    def as_pandas(self) -> pandas.DataFrame:
        return self._native.to_pandas()

    def rename(self, columns: Dict[str, str]) -> DataFrame:
        try:
            schema = self.schema.rename(columns)
        except Exception as e:
            raise FugueDataFrameOperationError from e
        df = self.native.rename(columns=columns)
        return CudaDataFrame(df, schema, type_safe=False)

    def alter_columns(self, columns: Any) -> DataFrame:
        new_schema = self._get_altered_schema(columns)
        if new_schema == self.schema:
            return self
        new_pdf = self.native.assign()
        for k, v in new_schema.items():
            if not v.type.equals(self.schema[k].type):
                old_type = self.schema[k].type
                new_type = v.type
                # int -> str
                if pa.types.is_integer(old_type) and pa.types.is_string(new_type):
                    series = new_pdf[k]
                    ns = series.isnull()
                    series = series.fillna(0).astype(int).astype(str)
                    new_pdf[k] = series.mask(ns, None)
                # bool -> str
                elif pa.types.is_boolean(old_type) and pa.types.is_string(new_type):
                    series = new_pdf[k]
                    ns = series.isnull()
                    positive = series != 0
                    new_pdf[k] = new_pdf[k].astype(str)
                    new_pdf[k] = "False"
                    new_pdf[k] = new_pdf[k].mask(positive, "True").mask(ns, None)
                # date -> str
                elif pa.types.is_date(old_type) and pa.types.is_string(new_type):
                    new_pdf[k] = new_pdf[k].dt.strftime("%Y-%m-%d")
                # datetime -> str
                elif pa.types.is_timestamp(old_type) and pa.types.is_string(new_type):
                    new_pdf[k] = new_pdf[k].dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    series = new_pdf[k]
                    ns = series.isnull()
                    series = series.astype(str)
                    new_pdf[k] = series.mask(ns, None)
        return CudaDataFrame(new_pdf, new_schema, type_safe=True)

    def as_array(
        self, columns: Optional[List[str]] = None, type_safe: bool = False
    ) -> List[Any]:
        # sdf = self._native if columns is None else self._native[columns]
        # adf = ArrowDataFrame(sdf.to_arrow(preserve_index=False))
        # if not type_safe:
        #     return adf.as_array(type_safe=type_safe)
        # schema = self.schema if columns is None else self.schema.extract(columns)
        # return ArrayDataFrame(adf.as_array(type_safe=False), schema).as_array(
        #     type_safe=True
        # )
        return list(self.as_array_iterable(columns=columns, type_safe=type_safe))

    def as_array_iterable(
        self, columns: Optional[List[str]] = None, type_safe: bool = False
    ) -> Iterable[Any]:
        return PD_UTILS.as_array_iterable(
            self.native.to_pandas(),
            schema=self.schema.pa_schema,
            columns=columns,
            type_safe=type_safe,
        )

    def head(self, n: int, columns: Optional[List[str]] = None) -> List[Any]:
        """Get first n rows of the dataframe as 2-dimensional array
        :param n: number of rows
        :param columns: selected columns, defaults to None (all columns)
        :return: 2-dimensional array
        """
        tdf = (
            self.native.head(n).to_pandas()
            if columns is None
            else self.native[columns].head(n).to_pandas()
        )
        schema = (
            self.schema.pa_schema
            if columns is None
            else self.schema.extract(columns).pa_schema
        )
        res: List[Any] = []

        for row in PD_UTILS.as_array_iterable(tdf, schema=schema, type_safe=True):
            if n < 1:
                break
            res.append(list(row))
            n -= 1
        return res

    def _apply_schema(
        self, pdf: cudf.DataFrame, schema: Optional[Schema]
    ) -> Tuple[cudf.DataFrame, Schema]:
        # CUDF_UTILS.ensure_compatible(pdf)
        if pdf.columns.dtype == "object":  # pdf has named schema
            pschema = Schema(CUDF_UTILS.to_schema(pdf))
            if schema is None or pschema == schema:
                return pdf, pschema.assert_not_empty()
            pdf = pdf[schema.assert_not_empty().names]
        else:  # pdf has no named schema
            schema = _input_schema(schema).assert_not_empty()
            assert_or_throw(
                pdf.shape[1] == len(schema),
                ValueError(f"Pandas datafame column count doesn't match {schema}"),
            )
            pdf.columns = schema.names
        return CUDF_UTILS.enforce_type(pdf, schema.pa_schema, null_safe=True), schema
