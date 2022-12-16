from typing import List, cast

from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import StructType

from spark_frame.utils import quote_columns


def flatten(df: DataFrame, struct_separator: str = ".") -> DataFrame:
    """Flatten all the struct columns of a Spark [DataFrame][pyspark.sql.DataFrame].
    Nested fields names will be joined together using the specified separator

    Args:
        df: A Spark DataFrame
        struct_separator: A string used to separate the structs names from their elements.
            It might be useful to change the separator when some DataFrame's column names already contain dots

    Returns:
        A flattened DataFrame

    Examples:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.createDataFrame(
        ...         [(1, {"a": 1, "b": {"c": 1, "d": 1}})],
        ...         "id INT, s STRUCT<a:INT, b:STRUCT<c:INT, d:INT>>"
        ...      )
        >>> df.printSchema()
        root
         |-- id: integer (nullable = true)
         |-- s: struct (nullable = true)
         |    |-- a: integer (nullable = true)
         |    |-- b: struct (nullable = true)
         |    |    |-- c: integer (nullable = true)
         |    |    |-- d: integer (nullable = true)
        <BLANKLINE>
        >>> flatten(df).printSchema()
        root
         |-- id: integer (nullable = true)
         |-- s.a: integer (nullable = true)
         |-- s.b.c: integer (nullable = true)
         |-- s.b.d: integer (nullable = true)
        <BLANKLINE>
        >>> df = spark.createDataFrame(
        ...         [(1, {"a.a1": 1, "b.b1": {"c.c1": 1, "d.d1": 1}})],
        ...         "id INT, `s.s1` STRUCT<`a.a1`:INT, `b.b1`:STRUCT<`c.c1`:INT, `d.d1`:INT>>"
        ... )
        >>> df.printSchema()
        root
         |-- id: integer (nullable = true)
         |-- s.s1: struct (nullable = true)
         |    |-- a.a1: integer (nullable = true)
         |    |-- b.b1: struct (nullable = true)
         |    |    |-- c.c1: integer (nullable = true)
         |    |    |-- d.d1: integer (nullable = true)
        <BLANKLINE>
        >>> flatten(df, "?").printSchema()
        root
         |-- id: integer (nullable = true)
         |-- s.s1?a.a1: integer (nullable = true)
         |-- s.s1?b.b1?c.c1: integer (nullable = true)
         |-- s.s1?b.b1?d.d1: integer (nullable = true)
        <BLANKLINE>

    """
    # The idea is to recursively write a "SELECT s.b.c as `s.b.c`" for each nested column.
    cols = []

    def expand_struct(struct: StructType, col_stack: List[str]) -> None:
        for field in struct:
            if type(field.dataType) == StructType:
                struct_field = cast(StructType, field.dataType)
                expand_struct(struct_field, col_stack + [field.name])
            else:
                column = f.col(".".join(quote_columns(col_stack + [field.name])))
                cols.append(column.alias(struct_separator.join(col_stack + [field.name])))

    expand_struct(df.schema, col_stack=[])
    return df.select(cols)
