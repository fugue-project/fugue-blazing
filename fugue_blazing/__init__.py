# flake8: noqa

import cudf
from fugue_blazing.dataframe import CudaDataFrame
from fugue_blazing.execution_engine import BlazingSQLEngine, CudaExecutionEngine

from fugue import register_execution_engine, register_sql_engine
from fugue.workflow import register_raw_df_type


def setup_shortcuts():
    register_raw_df_type(cudf.DataFrame)
    register_execution_engine("blazing", lambda conf: CudaExecutionEngine(conf))
    register_sql_engine("bsql", lambda engine: BlazingSQLEngine(engine, to_local=False))
    register_sql_engine(
        "local_bsql", lambda engine: BlazingSQLEngine(engine, to_local=True)
    )


setup_shortcuts()
