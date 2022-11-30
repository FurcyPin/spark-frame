from typing import Dict, List, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import StructType

from spark_frame.utils import quote


def parse_json_columns(df: DataFrame, columns: Union[str, List[str], Dict[str, str]]) -> DataFrame:
    """Transform the specified columns containing json strings in the given DataFrame into structs containing
    the equivalent parsed information.

    This method is similar to Spark's `from_json` function, with one main difference: `from_json` requires the user
    to pass the expected json schema, while this method performs a first pass on the DataFrame to detect automatically
    the json schema of each column.

    By default, the output columns will have the same name as the input columns, but if you want to keep the input
    columns you can pass a dict(input_col_name, output_col_name) to specify different output column names.

    Please be aware that automatic schema detection is not very robust, and while this method can be quite helpful
    for quick prototyping and data exploration, it is recommended to use a fixed schema and make sure the schema
    of the input json data is properly enforce, or at the very least use schema have a drift detection mechanism.

    WARNING : when you use this method on a column that is inside a struct (e.g. column "a.b.c"),
    instead of replacing that column it will create a new column outside the struct (e.g. "`a.b.c`") (See Example 2).

    Args:
        df: A Spark DataFrame
        columns: A column name, list of column names, or dict(column_name, parsed_column_name)

    Returns:
        A new DataFrame

    Examples:

        **Example 1 :**
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.createDataFrame([
        ...         (1, '[{"a": 1}, {"a": 2}]'),
        ...         (1, '[{"a": 2}, {"a": 4}]'),
        ...         (2, None)
        ...     ], "id INT, json1 STRING"
        ... )
        >>> df.show()
        +---+--------------------+
        | id|               json1|
        +---+--------------------+
        |  1|[{"a": 1}, {"a": 2}]|
        |  1|[{"a": 2}, {"a": 4}]|
        |  2|                null|
        +---+--------------------+
        <BLANKLINE>
        >>> df.printSchema()
        root
         |-- id: integer (nullable = true)
         |-- json1: string (nullable = true)
        <BLANKLINE>
        >>> parse_json_columns(df, 'json1').printSchema()
        root
         |-- id: integer (nullable = true)
         |-- json1: array (nullable = true)
         |    |-- element: struct (containsNull = true)
         |    |    |-- a: long (nullable = true)
        <BLANKLINE>

        **Example 2 : different output column name :**
        >>> parse_json_columns(df, {'json1': 'parsed_json1'}).printSchema()
        root
         |-- id: integer (nullable = true)
         |-- json1: string (nullable = true)
         |-- parsed_json1: array (nullable = true)
         |    |-- element: struct (containsNull = true)
         |    |    |-- a: long (nullable = true)
        <BLANKLINE>

        **Example 3 : json inside a struct :**
        >>> df = spark.createDataFrame([
        ...         (1, {'json1': '[{"a": 1}, {"a": 2}]'}),
        ...         (1, {'json1': '[{"a": 2}, {"a": 4}]'}),
        ...         (2, None)
        ...     ], "id INT, struct STRUCT<json1: STRING>"
        ... )
        >>> df.show(10, False)
        +---+----------------------+
        |id |struct                |
        +---+----------------------+
        |1  |{[{"a": 1}, {"a": 2}]}|
        |1  |{[{"a": 2}, {"a": 4}]}|
        |2  |null                  |
        +---+----------------------+
        <BLANKLINE>
        >>> df.printSchema()
        root
         |-- id: integer (nullable = true)
         |-- struct: struct (nullable = true)
         |    |-- json1: string (nullable = true)
        <BLANKLINE>
        >>> res = parse_json_columns(df, 'struct.json1')
        >>> res.printSchema()
        root
         |-- id: integer (nullable = true)
         |-- struct: struct (nullable = true)
         |    |-- json1: string (nullable = true)
         |-- struct.json1: array (nullable = true)
         |    |-- element: struct (containsNull = true)
         |    |    |-- a: long (nullable = true)
        <BLANKLINE>
        >>> res.show(10, False)
        +---+----------------------+------------+
        |id |struct                |struct.json1|
        +---+----------------------+------------+
        |1  |{[{"a": 1}, {"a": 2}]}|[{1}, {2}]  |
        |1  |{[{"a": 2}, {"a": 4}]}|[{2}, {4}]  |
        |2  |null                  |null        |
        +---+----------------------+------------+
        <BLANKLINE>

    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(columns, list):
        columns = {col: col for col in columns}

    wrapped_df = __wrap_json_columns(df, columns)
    schema_per_col = __infer_schema_per_column(wrapped_df, list(columns.values()))
    res = __parse_json_columns(wrapped_df, schema_per_col)
    return res


def _wrap_json(col: str, parsed_col: str) -> Column:
    """Transforms a string column into the json string '{"col_name": col_value}'

    It is necessary to wrap the json because the schema inference is incorrect for json that are arrays

    Examples:
    df.show()
    +---+--------------------+
    | id|               json1|
    +---+--------------------+
    |  1|[{"a": 1}, {"a": 2}]|
    |  1|[{"a": 2}, {"a": 4}]|
    +---+--------------------+
    schema = job.spark.read.json(df.select('json1').rdd.map(lambda r: r[0])).schema
    schema.simpleString()
    returns 'struct<a:bigint>' but the schema we need is 'array<struct<a:bigint>>'

    :param col:
    :return:
    """
    return f.concat(f.lit('{"'), f.lit(parsed_col), f.lit('": '), f.coalesce(col, f.lit("null")), f.lit("}"))


def __wrap_json_columns(df: DataFrame, columns: Dict[str, str]) -> DataFrame:
    """Wrap a json column inside a struct.

    >>> from pyspark.sql import SparkSession
    >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
    >>> df = spark.createDataFrame([
    ...         (1, '[{"a": 1}, {"a": 2}]'),
    ...         (1, '[{"a": 2}, {"a": 4}]'),
    ...         (2, None)
    ...     ], "id INT, json1 STRING"
    ... )
    >>> df.show()
    +---+--------------------+
    | id|               json1|
    +---+--------------------+
    |  1|[{"a": 1}, {"a": 2}]|
    |  1|[{"a": 2}, {"a": 4}]|
    |  2|                null|
    +---+--------------------+
    <BLANKLINE>
    >>> __wrap_json_columns(df, {"json1": "parsed_json1"}).show(truncate=False)
    +---+--------------------+--------------------------------------+
    |id |json1               |parsed_json1                          |
    +---+--------------------+--------------------------------------+
    |1  |[{"a": 1}, {"a": 2}]|{"parsed_json1": [{"a": 1}, {"a": 2}]}|
    |1  |[{"a": 2}, {"a": 4}]|{"parsed_json1": [{"a": 2}, {"a": 4}]}|
    |2  |null                |{"parsed_json1": null}                |
    +---+--------------------+--------------------------------------+
    <BLANKLINE>

    """
    res = df
    for col, parsed_col in columns.items():
        res = res.withColumn(parsed_col, _wrap_json(col, parsed_col))
    return res


def __infer_schema_per_column(df: DataFrame, columns: List[str]) -> Dict[str, StructType]:
    """Infer the schema of all the specified columns

    >>> from pyspark.sql import SparkSession
    >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
    >>> df = spark.createDataFrame([
    ...         (1, '[{"a": 1}, {"a": 2}]', '{"parsed_json1": [{"a": 1}, {"a": 2}]}'),
    ...         (1, '[{"a": 2}, {"a": 4}]', '{"parsed_json1": [{"a": 2}, {"a": 4}]}'),
    ...         (2, None, '{"parsed_json1": null}'),
    ...     ], "id INT, json1 STRING, parsed_json1 STRING"
    ... )
    >>> df.show(truncate=False)
    +---+--------------------+--------------------------------------+
    |id |json1               |parsed_json1                          |
    +---+--------------------+--------------------------------------+
    |1  |[{"a": 1}, {"a": 2}]|{"parsed_json1": [{"a": 1}, {"a": 2}]}|
    |1  |[{"a": 2}, {"a": 4}]|{"parsed_json1": [{"a": 2}, {"a": 4}]}|
    |2  |null                |{"parsed_json1": null}                |
    +---+--------------------+--------------------------------------+
    <BLANKLINE>
    >>> __infer_schema_per_column(df, ["parsed_json1"])["parsed_json1"].simpleString()
    'struct<parsed_json1:array<struct<a:bigint>>>'

    """
    spark = df.sparkSession
    schema_per_col = {}
    for col in columns:
        schema = spark.read.json(df.select(quote(col)).rdd.map(lambda r: r[0])).schema
        schema_per_col[col] = schema
    return schema_per_col


def __parse_json_columns(df: DataFrame, schema_per_col: Dict[str, StructType]) -> DataFrame:
    """Infer the schema of all the specified columns

    >>> from pyspark.sql import SparkSession
    >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
    >>> df = spark.createDataFrame([
    ...         (1, '[{"a": 1}, {"a": 2}]', '{"parsed_json1": [{"a": 1}, {"a": 2}]}'),
    ...         (1, '[{"a": 2}, {"a": 4}]', '{"parsed_json1": [{"a": 2}, {"a": 4}]}'),
    ...         (2, None, '{"parsed_json1": null}'),
    ...     ], "id INT, json1 STRING, parsed_json1 STRING"
    ... )
    >>> df.show(truncate=False)
    +---+--------------------+--------------------------------------+
    |id |json1               |parsed_json1                          |
    +---+--------------------+--------------------------------------+
    |1  |[{"a": 1}, {"a": 2}]|{"parsed_json1": [{"a": 1}, {"a": 2}]}|
    |1  |[{"a": 2}, {"a": 4}]|{"parsed_json1": [{"a": 2}, {"a": 4}]}|
    |2  |null                |{"parsed_json1": null}                |
    +---+--------------------+--------------------------------------+
    <BLANKLINE>
    >>> from spark_frame.schema_utils import schema_from_simple_string
    >>> schema = schema_from_simple_string('struct<parsed_json1:array<struct<a:bigint>>>')
    >>> res = __parse_json_columns(df, {"parsed_json1": schema})
    >>> res.printSchema()
    root
     |-- id: integer (nullable = true)
     |-- json1: string (nullable = true)
     |-- parsed_json1: array (nullable = true)
     |    |-- element: struct (containsNull = true)
     |    |    |-- a: long (nullable = true)
    <BLANKLINE>
    >>> res.show(truncate=False)
    +---+--------------------+------------+
    |id |json1               |parsed_json1|
    +---+--------------------+------------+
    |1  |[{"a": 1}, {"a": 2}]|[{1}, {2}]  |
    |1  |[{"a": 2}, {"a": 4}]|[{2}, {4}]  |
    |2  |null                |null        |
    +---+--------------------+------------+
    <BLANKLINE>

    """
    res = df
    for parsed_col, schema in schema_per_col.items():
        res = res.withColumn(parsed_col, f.from_json(quote(parsed_col), schema).alias(parsed_col)[parsed_col])
    return res
