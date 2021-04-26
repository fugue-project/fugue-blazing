# Fugue, Rapids, BlazingSQL integration

[![PyPI version](https://badge.fury.io/py/fugue-blazing.svg)](https://pypi.python.org/pypi/fugue-blazing/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/fugue-blazing.svg)](https://pypi.python.org/pypi/fugue-blazing/)
[![PyPI license](https://img.shields.io/pypi/l/fugue-blazing.svg)](https://pypi.python.org/pypi/fugue-blazing/)
[![Doc](https://readthedocs.org/projects/fugue-blazing/badge)](https://fugue-blazing.readthedocs.org)

[![Slack Status](https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social)](https://join.slack.com/t/fugue-project/shared_invite/zt-jl0pcahu-KdlSOgi~fP50TZWmNxdWYQ)


This project extends [Fugue](https://github.com/fugue-project/fugue) to support [Rapids](https://rapids.ai/index.html) [cuDF](https://docs.rapids.ai/api/cudf/stable/) and [BlazingSQL](https://blazingsql.com/).

## Installation

You need to install Rapids and BlazingSQL by yourself (see [official instructions](https://rapids.ai/start.html#get-rapids)), and assume you installed them by conda, then you need to pip install in the same environment

```bash
conda run -n <your_env> pip install fugue-blazing
```

## How To Use

As a standard Fugue extension, you can use in two ways: [functional APIs](https://fugue-tutorials.readthedocs.io/en/latest/README.html) and [Fugue SQL](https://fugue-tutorials.readthedocs.io/en/latest/tutorials/sql.html). But Fugue SQL is the preferred way for this extension. This is because due to the special design of GPU, code to run on GPU has special requirement. Currently [transform](https://fugue-tutorials.readthedocs.io/en/latest/tutorials/transformer.html) is leveraging [NativeExecutionEngine](https://fugue.readthedocs.io/en/latest/api/fugue.execution.html#module-fugue.execution.native_execution_engine) which is using CPU. Other than transform, Fugue fully relies on cuDF and BlasingSQL to do the compute.

Practically, if you don't use transform, then SQL may be the better choice to express your data pipelines.

### Functional APIs

Here is an example Fugue code snippet that illustrates some of the key features of the framework. A fillna function creates a new column named `filled`, which is the same as the column `value` except that the `None` values are filled.

```python
from fugue import FugueWorkflow
from fugue_blazing import CudaExecutionEngine, setup_shortcuts

# Creating sample data
data = [
    ["A", "2020-01-01", 10],
    ["A", "2020-01-02", None],
    ["A", "2020-01-03", 30],
    ["B", "2020-01-01", 20],
    ["B", "2020-01-02", None],
    ["B", "2020-01-03", 40]
]
schema = "id:str,date:date,value:double"

dag = FugueWorkflow()
dag.df(data, schema).partition_by("id", presort="date").take(1).show()

dag.run(CudaExecutionEngine)

# call setup_shortcuts to make your code more expressive
setup_shortcuts()
dag.run("blazing")
```

You can also run SQL using functional API:

```python
from fugue import FugueWorkflow
from fugue_blazing import setup_shortcuts

setup_shortcuts()

data = [
    ["A", "2020-01-01", 10],
    ["A", "2020-01-02", None],
    ["A", "2020-01-03", 30],
    ["B", "2020-01-01", 20],
    ["B", "2020-01-02", None],
    ["B", "2020-01-03", 40]
]
schema = "id:str,date:date,value:double"

with FugueWorkflow("blazing") as dag:
    df = dag.df(data, schema)
    dag.select("* from ",df," where value>20").show()
```

For detailed examples, please read [Fugue Tutorials](https://fugue.readthedocs.io/en/latest/tutorials.html)


### Fugue SQL

#### Programmatical Approach

```python
from fugue_sql import fsql
from fugue_blazing import setup_shortcuts
import pandas as pd
import cudf

setup_shortcuts()

pdf = pd.DataFrame([
    ["A", "2020-01-01", 10],
    ["A", "2020-01-02", None],
    ["A", "2020-01-03", 30],
    ["B", "2020-01-01", 20],
    ["B", "2020-01-02", None],
    ["B", "2020-01-03", 40]
], columns = ["id", "date", "value"])

result = fsql("""
TAKE 1 ROW FROM df PREPARTITION BY id PRESORT date
YIELD DATAFRAME AS x
""", df=pdf).run("blazing")

# this is how you get outputs from Fugue SQL
assert isinstance(result["x"].native, cudf.DataFrame)

fsql("""
SELECT * FROM best WHERE id='A'
PRINT
SELECT id, COUNT(*) AS ct FROM orig GROUP BY id
PRINT
""", best=result["x"], orig=pdf).run("blazing")
```


#### Jupyter Notebook


Before running Jupyter, you need to firstly install fugue and notebook extension

```bash
pip install fugue
jupyter nbextension install --sys-prefix --symlink --py fugue_notebook
jupyter nbextension enable --py fugue_notebook
```

In cell 1

```python
%load_ext fugue_notebook

from fugue_blazing import setup_shortcuts
setup_shortcuts()

pdf = pd.DataFrame([
    ["A", "2020-01-01", 10],
    ["A", "2020-01-02", None],
    ["A", "2020-01-03", 30],
    ["B", "2020-01-01", 20],
    ["B", "2020-01-02", None],
    ["B", "2020-01-03", 40]
], columns = ["id", "date", "value"])
```

In cell 2

```bash
%%fsql blazing
TAKE 1 ROW FROM df PREPARTITION BY id PRESORT date
YIELD DATAFRAME AS x
```

In cell 3

```bash
%%fsql blazing
SELECT * FROM x WHERE id='A'
PRINT
SELECT id, COUNT(*) AS ct FROM pdf GROUP BY id
PRINT
```
