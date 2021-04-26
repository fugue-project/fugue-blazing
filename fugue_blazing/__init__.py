# flake8: noqa

import cudf
from fugue_blazing.dataframe import CudaDataFrame
from fugue_blazing.execution_engine import CudaExecutionEngine

from fugue.workflow import register_raw_df_type
from fugue import register_execution_engine


def setup_shortcuts():
    register_raw_df_type(cudf.DataFrame)
    register_execution_engine("blazing", lambda conf: CudaExecutionEngine(conf))


setup_shortcuts()
