import json
import math
from datetime import date, datetime
from typing import Any

import cudf
import numpy as np
import pandas as pd
from fugue_blazing.dataframe import CudaDataFrame
from fugue_test.dataframe_suite import DataFrameTests
from pytest import raises
from triad.collections.schema import Schema

from fugue import ArrayDataFrame, ArrowDataFrame, PandasDataFrame
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import FugueDataFrameInitError


class CudaDataFrameTests(DataFrameTests.Tests):
    def df(
        self, data: Any = None, schema: Any = None, metadata: Any = None
    ) -> CudaDataFrame:
        return CudaDataFrame(data, schema, metadata)

    def test_as_arrow(self):
        return

    def test_as_arrow_2(self):
        # empty
        df = self.df([], "a:int,b:int")
        assert [] == list(ArrowDataFrame(df.as_arrow()).as_dict_iterable())
        # pd.Nat
        df = self.df([[pd.NaT, 1]], "a:datetime,b:int")
        assert [dict(a=None, b=1)] == list(
            ArrowDataFrame(df.as_arrow()).as_dict_iterable()
        )
        # pandas timestamps
        df = self.df([[pd.Timestamp("2020-01-01"), 1]], "a:datetime,b:int")
        assert [dict(a=datetime(2020, 1, 1), b=1)] == list(
            ArrowDataFrame(df.as_arrow()).as_dict_iterable()
        )
        # float nan, list
        data = [[[float("nan"), 2.0]]]
        df = self.df(data, "a:[float]")
        assert [[[None, 2.0]]] == ArrowDataFrame(df.as_arrow()).as_array()
        # dict
        data = [[dict(b="x")]]
        df = self.df(data, "a:{b:str}")
        assert data == ArrowDataFrame(df.as_arrow()).as_array()

        # TODO: nested type is not handled correctly by cudf
        # list[dict]
        # data = [[[dict(b=[30, 40])]]]
        # df = self.df(data, "a:[{b:[int]}]")
        # assert data == ArrowDataFrame(df.as_arrow()).as_array()

    def test_nested(self):
        return

    def test_nested_2(self):
        data = [[[30, 40]]]
        df = self.df(data, "a:[int]")
        a = df.as_array(type_safe=True)
        assert data == a

        data = [[dict(a="1", b=[3, 4], d=1.0)], [dict(b=[30, 40])]]
        df = self.df(data, "a:{a:str,b:[int]}")
        a = df.as_array(type_safe=True)
        assert [[dict(a="1", b=[3, 4])], [dict(a=None, b=[30, 40])]] == a

        # TODO: nested type is not fully supported by cudf
        # data = [[[dict(b=[30, 40])]]]
        # df = self.df(data, "a:[{a:str,b:[int]}]")
        # a = df.as_array(type_safe=True)
        # assert [[[dict(a=None, b=[30, 40])]]] == a

    def test_binary(self) -> None:  # CudaDataFrame does not support binary
        return

    def test_alter_columns(self):
        return

    def test_alter_columns_2(self):
        # empty
        df = self.df([], "a:str,b:int")
        ndf = df.alter_columns("a:str,b:str")
        assert [] == ndf.as_array(type_safe=True)
        assert ndf.schema == "a:str,b:str"

        # no change
        df = self.df([["a", 1], ["c", None]], "a:str,b:int")
        ndf = df.alter_columns("b:int,a:str")
        assert [["a", 1], ["c", None]] == ndf.as_array(type_safe=True)
        assert ndf.schema == df.schema

        # bool -> str
        df = self.df([["a", True], ["b", False], ["c", None]], "a:str,b:bool")
        ndf = df.alter_columns("b:str")
        actual = ndf.as_array(type_safe=True)
        # Capitalization doesn't matter
        # and dataframes don't need to be consistent on capitalization
        expected1 = [["a", "True"], ["b", "False"], ["c", None]]
        expected2 = [["a", "true"], ["b", "false"], ["c", None]]
        assert expected1 == actual or expected2 == actual
        assert ndf.schema == "a:str,b:str"

        # int -> str
        df = self.df([["a", 1], ["c", None]], "a:str,b:int")
        ndf = df.alter_columns("b:str")
        assert [["a", "1"], ["c", None]] == ndf.as_array(type_safe=True)
        assert ndf.schema == "a:str,b:str"

        # double -> str
        df = self.df([["a", 1.1], ["b", None]], "a:str,b:double")
        data = df.alter_columns("b:str").as_array(type_safe=True)
        assert [["a", "1.1"], ["b", None]] == data

        # date -> str
        df = self.df(
            [["a", date(2020, 1, 1)], ["b", date(2020, 1, 2)], ["c", None]],
            "a:str,b:date",
        )
        data = df.alter_columns("b:str").as_array(type_safe=True)
        assert [["a", "2020-01-01"], ["b", "2020-01-02"], ["c", None]] == data

        # datetime -> str
        df = self.df(
            [
                ["a", datetime(2020, 1, 1, 3, 4, 5)],
                ["b", datetime(2020, 1, 2, 16, 7, 8)],
                ["c", None],
            ],
            "a:str,b:datetime",
        )
        data = df.alter_columns("b:str").as_array(type_safe=True)
        assert [
            ["a", "2020-01-01 03:04:05"],
            ["b", "2020-01-02 16:07:08"],
            ["c", None],
        ] == data

        # str -> bool
        df = self.df([["a", "trUe"], ["b", "False"], ["c", None]], "a:str,b:str")
        ndf = df.alter_columns("b:bool,a:str")
        assert [["a", True], ["b", False], ["c", None]] == ndf.as_array(type_safe=True)
        assert ndf.schema == "a:str,b:bool"

        # str -> int
        df = self.df([["a", "1"]], "a:str,b:str")
        ndf = df.alter_columns("b:int,a:str")
        assert [["a", 1]] == ndf.as_array(type_safe=True)
        assert ndf.schema == "a:str,b:int"

        # str -> double
        df = self.df([["a", "1.1"], ["b", "2"], ["c", None]], "a:str,b:str")
        ndf = df.alter_columns("b:double")
        for x, y in zip(
            [["a", 1.1], ["b", 2.0], ["c", None]], ndf.as_array(type_safe=True)
        ):
            assert x[0] == y[0]
            assert (x[1] is None and y[1] is None) or abs(x[1] - y[1]) < 0.0001
        assert ndf.schema == "a:str,b:double"

        # str -> date
        df = self.df(
            [["1", "2020-01-01"], ["2", "2020-01-02 01:02:03"], ["3", None]],
            "a:str,b:str",
        )
        ndf = df.alter_columns("b:date,a:int")
        assert [
            [1, date(2020, 1, 1)],
            [2, date(2020, 1, 2)],
            [3, None],
        ] == ndf.as_array(type_safe=True)
        assert ndf.schema == "a:int,b:date"

        # str -> datetime
        # cuda can't mix date and datetime
        # df = self.df(
        #     [["1", "2020-01-01"], ["2", "2020-01-02 01:02:03"], ["3", None]],
        #     "a:str,b:str",
        # )
        # ndf = df.alter_columns("b:datetime,a:int")
        # assert [
        #     [1, datetime(2020, 1, 1)],
        #     [2, datetime(2020, 1, 2, 1, 2, 3)],
        #     [3, None],
        # ] == ndf.as_array(type_safe=True)
        # assert ndf.schema == "a:int,b:datetime"

        df = self.df(
            [["1", "2020-01-01 16:17:18"], ["2", "2020-01-02 01:02:03"], ["3", None]],
            "a:str,b:str",
        )
        ndf = df.alter_columns("b:datetime,a:int")
        assert [
            [1, datetime(2020, 1, 1, 16, 17, 18)],
            [2, datetime(2020, 1, 2, 1, 2, 3)],
            [3, None],
        ] == ndf.as_array(type_safe=True)
        assert ndf.schema == "a:int,b:datetime"


def test_init():
    df = CudaDataFrame(schema="a:str,b:int")
    assert df.is_bounded
    assert df.count() == 0
    assert df.schema == "a:str,b:int"

    pdf = pd.DataFrame([["a", 1], ["b", 2]])
    raises(FugueDataFrameInitError, lambda: CudaDataFrame(pdf))
    df = CudaDataFrame(pdf, "a:str,b:str")
    assert [["a", "1"], ["b", "2"]] == df.as_pandas().values.tolist()
    df = CudaDataFrame(pdf, "a:str,b:int")
    assert [["a", 1], ["b", 2]] == df.as_pandas().values.tolist()
    df = CudaDataFrame(pdf, "a:str,b:double")
    assert [["a", 1.0], ["b", 2.0]] == df.as_pandas().values.tolist()

    pdf = CudaDataFrame([["a", 1], ["b", 2]], "a:str,b:int").native["b"]
    assert isinstance(pdf, cudf.Series)
    df = CudaDataFrame(pdf, "b:str")
    assert [["1"], ["2"]] == df.as_pandas().values.tolist()
    df = CudaDataFrame(pdf, "b:double")
    assert [[1.0], [2.0]] == df.as_pandas().values.tolist()

    pdf = CudaDataFrame([["a", 1], ["b", 2]], "x:str,y:long").native
    df = CudaDataFrame(pdf)
    assert df.schema == "x:str,y:long"
    df = CudaDataFrame(pdf, "y:str,x:str")
    assert [["1", "a"], ["2", "b"]] == df.as_pandas().values.tolist()
    ddf = CudaDataFrame(df)
    assert [["1", "a"], ["2", "b"]] == ddf.as_pandas().values.tolist()
    assert df.native is ddf.native  # no real copy happened

    df = CudaDataFrame([["a", 1], ["b", "2"]], "x:str,y:double")
    assert [["a", 1.0], ["b", 2.0]] == df.as_pandas().values.tolist()

    df = CudaDataFrame([], "x:str,y:double")
    assert [] == df.as_pandas().values.tolist()

    raises(FugueDataFrameInitError, lambda: CudaDataFrame(123))


def test_simple_methods():
    df = CudaDataFrame([], "a:str,b:int")
    assert df.empty
    assert 0 == df.count()
    assert not df.is_local

    df = CudaDataFrame([["a", 1], ["b", "2"]], "x:str,y:double")
    assert not df.empty
    assert 2 == df.count()
    assert ["a", 1.0] == df.peek_array()
    assert dict(x="a", y=1.0) == df.peek_dict()

    df_eq(
        PandasDataFrame(df.as_pandas()),
        [["a", 1.0], ["b", 2.0]],
        "x:str,y:double",
        throw=True,
    )


def _test_nested():
    # TODO: nested type doesn't work in cudf
    # data = [[dict(a=1, b=[3, 4], d=1.0)], [json.dumps(dict(b=[30, "40"]))]]
    # df = CudaDataFrame(data, "a:{a:str,b:[int]}")
    # a = df.as_array(type_safe=True)
    # assert [[dict(a="1", b=[3, 4])], [dict(a=None, b=[30, 40])]] == a

    data = [[[json.dumps(dict(b=[30, "40"]))]]]
    df = CudaDataFrame(data, "a:[{a:str,b:[int]}]")
    a = df.as_array(type_safe=True)
    assert [[[dict(a=None, b=[30, 40])]]] == a


def test_as_array():
    df = CudaDataFrame([], "a:str,b:int")
    assert [] == df.as_array()
    assert [] == df.as_array(type_safe=True)
    assert [] == list(df.as_array_iterable())
    assert [] == list(df.as_array_iterable(type_safe=True))

    df = CudaDataFrame([["a", 1]], "a:str,b:int")
    assert [["a", 1]] == df.as_array()
    assert [["a", 1]] == df.as_array(["a", "b"])
    assert [[1, "a"]] == df.as_array(["b", "a"])

    # prevent pandas auto type casting
    df = CudaDataFrame([[1.0, 1.1]], "a:double,b:int")
    assert [[1.0, 1]] == df.as_array()
    assert isinstance(df.as_array()[0][0], float)
    assert isinstance(df.as_array()[0][1], int)
    assert [[1.0, 1]] == df.as_array(["a", "b"])
    assert [[1, 1.0]] == df.as_array(["b", "a"])

    df = CudaDataFrame([[np.float64(1.0), 1.1]], "a:double,b:int")
    assert [[1.0, 1]] == df.as_array()
    assert isinstance(df.as_array()[0][0], float)
    assert isinstance(df.as_array()[0][1], int)

    df = CudaDataFrame([[pd.Timestamp("2020-01-01"), 1.1]], "a:datetime,b:int")
    df.native["a"] = cudf.to_datetime(df.native["a"])
    assert [[datetime(2020, 1, 1), 1]] == df.as_array()
    assert isinstance(df.as_array()[0][0], datetime)
    assert isinstance(df.as_array()[0][1], int)

    df = CudaDataFrame([[pd.NaT, 1.1]], "a:datetime,b:int")
    df.native["a"] = cudf.to_datetime(df.native["a"])
    assert isinstance(df.as_array()[0][0], datetime)
    assert isinstance(df.as_array()[0][1], int)

    df = CudaDataFrame([[1.0, 1.1]], "a:double,b:int")
    assert [[1.0, 1]] == df.as_array(type_safe=True)
    assert isinstance(df.as_array()[0][0], float)
    assert isinstance(df.as_array()[0][1], int)


def test_as_dict_iterable():
    df = CudaDataFrame([["2020-01-01", 1.1]], "a:datetime,b:int")
    assert [dict(a=datetime(2020, 1, 1), b=1)] == list(df.as_dict_iterable())


def test_nan_none():
    # TODO: on dask, these tests can't pass
    # df = ArrayDataFrame([[None, None]], "b:str,c:double")
    # assert df.as_pandas().iloc[0, 0] is None
    # arr = CudaDataFrame(df.as_pandas(), df.schema).as_array()[0]
    # assert arr[0] is None
    # assert math.isnan(arr[1])

    # df = ArrayDataFrame([[None, None]], "b:int,c:bool")
    # arr = CudaDataFrame(df.as_pandas(), df.schema).as_array(type_safe=True)[0]
    # assert np.isnan(arr[0])  # TODO: this will cause inconsistent behavior cross engine
    # assert np.isnan(arr[1])  # TODO: this will cause inconsistent behavior cross engine

    df = ArrayDataFrame([["a", 1.1], [None, None]], "b:str,c:double")
    arr = CudaDataFrame(df.as_pandas(), df.schema).as_array()[1]
    assert arr[0] is None
    assert math.isnan(arr[1])

    arr = CudaDataFrame(df.as_array(), df.schema).as_array()[1]
    assert arr[0] is None
    assert math.isnan(arr[1])

    arr = CudaDataFrame(df.as_pandas()["b"], "b:str").as_array()[1]
    assert arr[0] is None


def _test_as_array_perf():
    s = Schema()
    arr = []
    for i in range(100):
        s.append(f"a{i}:int")
        arr.append(i)
    for i in range(100):
        s.append(f"b{i}:int")
        arr.append(float(i))
    for i in range(100):
        s.append(f"c{i}:str")
        arr.append(str(i))
    data = []
    for i in range(5000):
        data.append(list(arr))
    df = CudaDataFrame(data, s)
    res = df.as_array()
    res = df.as_array(type_safe=True)
    nts, ts = 0.0, 0.0
    for i in range(10):
        t = datetime.now()
        res = df.as_array()
        nts += (datetime.now() - t).total_seconds()
        t = datetime.now()
        res = df.as_array(type_safe=True)
        ts += (datetime.now() - t).total_seconds()
    print(nts, ts)
