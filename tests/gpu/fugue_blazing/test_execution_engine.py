import pandas as pd
from fugue_blazing.execution_engine import CudaExecutionEngine
from fugue_test.builtin_suite import BuiltInTests
from fugue_test.execution_suite import ExecutionEngineTests

from fugue import SqliteEngine
from fugue.collections.partition import PartitionSpec
from fugue.dataframe.pandas_dataframe import PandasDataFrame
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.workflow.workflow import FugueWorkflow


class CudaExecutionEngineTests(ExecutionEngineTests.Tests):
    def make_engine(self):
        e = CudaExecutionEngine(dict(test=True))
        return e

    def test__join_outer_pandas_incompatible(self):
        return

    def test_map_with_dict_col(self):
        # TODO: add back
        return

    def test_to_df(self):
        e = self.engine
        a = e.to_df([[1, 2], [3, 4]], "a:int,b:int", dict(a=1))
        df_eq(a, [[1, 2], [3, 4]], "a:int,b:int", dict(a=1), throw=True)
        a = e.to_df(PandasDataFrame([[1, 2], [3, 4]], "a:int,b:int", dict(a=1)))
        df_eq(a, [[1, 2], [3, 4]], "a:int,b:int", dict(a=1), throw=True)
        assert a is e.to_df(a)

    def test_intersect(self):
        return

    def test_subtract(self):
        return


class CudaExecutionEngineBuiltInTests(BuiltInTests.Tests):
    def make_engine(self):
        e = CudaExecutionEngine(dict(test=True))
        return e

    def test_intersect(self):
        return

    def test_subtract(self):
        return

    def test_transform_binary(self):
        return

    def test_default_init(self):
        a = FugueWorkflow().df([[0]], "a:int")
        df_eq(a.compute(CudaExecutionEngine), [[0]], "a:int")

    def test_select(self):
        class MockEngine(SqliteEngine):
            def __init__(self, execution_engine, p: int = 0):
                super().__init__(execution_engine)
                self.p = p

            def select(self, dfs, statement):
                assert 2 == self.p  # assert set p value is working
                return super().select(dfs, statement)

        # schema: *
        def mock_tf1(df: pd.DataFrame) -> pd.DataFrame:
            return df

        with self.dag() as dag:
            a = dag.df([[1, 10], [2, 20], [3, 30]], "x:long,y:long")
            b = dag.df([[2, 20, 40], [3, 30, 90]], "x:long,y:long,z:long")
            dag.select("* FROM", a).assert_eq(a)
            dag.select("SELECT *,x*y AS z FROM", a, "WHERE x>=2").assert_eq(b)

            c = dag.df([[2, 20, 40], [3, 30, 90]], "x:long,y:long,zb:long")
            dag.select(
                "  SELECT t1.*,z AS zb FROM ",
                a,
                "AS t1 INNER JOIN",
                b,
                "AS t2 ON t1.x=t2.x  ",
            ).assert_eq(c)

            # no select
            dag.select(
                "t1.*,z AS zb FROM ", a, "AS t1 INNER JOIN", b, "AS t2 ON t1.x=t2.x"
            ).assert_eq(c)

            # specify sql engine
            dag.select(
                "SELECT t1.*,z AS zb FROM ",
                a,
                "AS t1 INNER JOIN",
                b,
                "AS t2 ON t1.x=t2.x",
                sql_engine="sqlite",
            ).assert_eq(c)

            # specify sql engine and params
            dag.select(
                "SELECT t1.*,z AS zb FROM ",
                a,
                "AS t1 INNER JOIN",
                b,
                "AS t2 ON t1.x=t2.x",
                sql_engine=MockEngine,
                sql_engine_params={"p": 2},
            ).assert_eq(c)

            # Blazing SQL doesn't support this
            # no input
            # dag.select("9223372036854775807 AS a").assert_eq(
            #    dag.df([[9223372036854775807]], "a:long")
            # )

            # make sure transform -> select works
            b = a.transform(mock_tf1)
            a = a.transform(mock_tf1)
            aa = dag.select("* FROM", a)
            dag.select("* FROM", b).assert_eq(aa)
