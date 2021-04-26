from typing import Iterable, List

import cudf
import numpy as np
import pyarrow as pa
from qpd import PandasLikeUtils
from triad import assert_or_throw
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP


class CUDFUtils(PandasLikeUtils[cudf.DataFrame, cudf.Series]):
    def is_compatile_index(self, df: cudf.DataFrame) -> bool:
        """Check whether the datafame is compatible with the operations inside
        this utils collection

        :param df: pandas like dataframe
        :return: if it is compatible
        """
        return isinstance(
            df.index, (cudf.RangeIndex, cudf.Int64Index, cudf.UInt64Index)
        )

    def to_schema(self, df: cudf.DataFrame) -> pa.Schema:
        """Extract cudf schema as pyarrow schema. This is a replacement
        of pyarrow.Schema.from_pandas, and it can correctly handle string type and
        empty dataframes

        :param df: pandas dataframe
        :raises ValueError: if pandas dataframe does not have named schema
        :return: pyarrow.Schema

        :Notice:
        The dataframe must be either empty, or with type pd.RangeIndex, pd.Int64Index
        or pd.UInt64Index and without a name, otherwise, `ValueError` will raise.
        """
        self.ensure_compatible(df)
        assert_or_throw(
            df.columns.dtype == "object",
            ValueError("Pandas dataframe must have named schema"),
        )

        def get_fields() -> Iterable[pa.Field]:

            for i in range(df.shape[1]):
                tp = df.dtypes[i]
                if tp == np.dtype("object") or tp == np.dtype(str):
                    t = pa.string()
                elif isinstance(
                    tp, (cudf.core.dtypes.ListDtype, cudf.core.dtypes.StructDtype)
                ):
                    t = tp.to_arrow()
                else:
                    t = pa.from_numpy_dtype(tp)
                yield pa.field(df.columns[i], t)

        fields: List[pa.Field] = []
        for field in get_fields():
            if pa.types.is_timestamp(field.type):
                fields.append(pa.field(field.name, TRIAD_DEFAULT_TIMESTAMP))
            else:
                fields.append(field)
        return pa.schema(fields)


CUDF_UTILS = CUDFUtils()
