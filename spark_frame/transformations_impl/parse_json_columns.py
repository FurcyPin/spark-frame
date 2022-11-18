from typing import Dict, List, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

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

    Example 1 :
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

    Example 2 : different output column name:
    >>> parse_json_columns(df, {'json1': 'parsed_json1'}).printSchema()
    root
     |-- id: integer (nullable = true)
     |-- json1: string (nullable = true)
     |-- parsed_json1: array (nullable = true)
     |    |-- element: struct (containsNull = true)
     |    |    |-- a: long (nullable = true)
    <BLANKLINE>

    Example 3 : json inside a struct :
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

    :param df: a DataFrame
    :param columns: A column name, list of column names, or dict(column_name, parsed_column_name)
    :return: a new DataFrame
    """
    spark = df.sparkSession
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(columns, list):
        columns = {col: col for col in columns}

    def wrap_json(col: str, parsed_col: str) -> Column:
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

    res = df
    for col, parsed_col in columns.items():
        # >>> res.show()
        # +---+--------------------+
        # | id|               json1|
        # +---+--------------------+
        # |  1|[{"a": 1}, {"a": 2}]|
        # |  1|[{"a": 2}, {"a": 4}]|
        # |  2|                null|
        # +---+--------------------+
        res = res.withColumn(parsed_col, wrap_json(col, parsed_col))
        # >>> res.show(10, False)
        # +---+--------------------+--------------------------------------+
        # |id |json1               |parsed_json1                          |
        # +---+--------------------+--------------------------------------+
        # |1  |[{"a": 1}, {"a": 2}]|{"parsed_json1": [{"a": 1}, {"a": 2}]}|
        # |1  |[{"a": 2}, {"a": 4}]|{"parsed_json1": [{"a": 2}, {"a": 4}]}|
        # |2  |null                |{"parsed_json1": null}                |
        # +---+--------------------+--------------------------------------+
        schema = spark.read.json(res.select(quote(parsed_col)).rdd.map(lambda r: r[0])).schema
        # >>> schema.simpleString()
        # 'struct<parsed_json1:array<struct<a:bigint>>>'
        res = res.withColumn(parsed_col, f.from_json(quote(parsed_col), schema).alias(parsed_col)[parsed_col])
        # >>> res.printSchema()
        # root
        #  |-- id: integer (nullable = true)
        #  |-- json1: string (nullable = true)
        #  |-- parsed_json1: array (nullable = true)
        #  |    |-- element: struct (containsNull = true)
        #  |    |    |-- a: long (nullable = true)
        # >>> res.show()
        # +---+--------------------+------------+
        # | id|               json1|parsed_json1|
        # +---+--------------------+------------+
        # |  1|[{"a": 1}, {"a": 2}]|  [{1}, {2}]|
        # |  1|[{"a": 2}, {"a": 4}]|  [{2}, {4}]|
        # |  2|                null|        null|
        # +---+--------------------+------------+
    return res
