from typing import Union

from pyspark.sql import Column
from pyspark.sql import functions as f
from pyspark.sql.types import DataType, StringType

from spark_frame.utils import quote


def empty_array(element_type: Union[DataType, str]) -> Column:
    """Create an empty Spark array column of the specified type.
    This is a workaround to the Spark method `typedLit` not being available in PySpark

    Args:
        element_type: The type of the array's element

    Returns:
        A Spark Column representing an empty array.

    Examples:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> from pyspark.sql import functions as f
        >>> df = spark.sql('''SELECT 1 as a''')
        >>> res = df.withColumn('empty_array', empty_array("STRUCT<b: int, c: array<string>>"))
        >>> res.printSchema()
        root
         |-- a: integer (nullable = false)
         |-- empty_array: array (nullable = false)
         |    |-- element: struct (containsNull = true)
         |    |    |-- b: integer (nullable = true)
         |    |    |-- c: array (nullable = true)
         |    |    |    |-- element: string (containsNull = true)
        <BLANKLINE>
        >>> res.show()
        +---+-----------+
        |  a|empty_array|
        +---+-----------+
        |  1|         []|
        +---+-----------+
        <BLANKLINE>
    """
    return f.array_except(f.array(f.lit(None).cast(element_type)), f.array(f.lit(None)))


def generic_struct(*columns: str, col_name_alias: str = "name", col_value_alias: str = "value") -> Column:
    """Transform a set of columns into a generic array of struct of type ARRAY<STRUCT<name: STRING, value: STRING>
    (column_name -> column_value)

    Args:
        *columns: One or multiple column names to add to the generic struct
        col_name_alias: Alias of the field containing the column names in the returned struct
        col_value_alias: Alias of the field containing the column values in the returned struct

    Returns:
        A Spark Column

    Examples:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''
        ...     SELECT
        ...       col1 as `pokemon.id`,
        ...       col2 as `pokemon.name`,
        ...       col3 as `pokemon.types`
        ...     FROM VALUES
        ...       (4, 'Charmander', ARRAY(NAMED_STRUCT('type', 'Fire'))),
        ...       (5, 'Charmeleon', ARRAY(NAMED_STRUCT('type', 'Fire'))),
        ...       (6, 'Charizard',  ARRAY(NAMED_STRUCT('type', 'Fire'), NAMED_STRUCT('type', 'Flying')))
        ... ''')
        >>> df.show()
        +----------+------------+------------------+
        |pokemon.id|pokemon.name|     pokemon.types|
        +----------+------------+------------------+
        |         4|  Charmander|          [{Fire}]|
        |         5|  Charmeleon|          [{Fire}]|
        |         6|   Charizard|[{Fire}, {Flying}]|
        +----------+------------+------------------+
        <BLANKLINE>
        >>> res = df.select(generic_struct("pokemon.id", "pokemon.name", "pokemon.types").alias('values'))
        >>> res.printSchema()
        root
         |-- values: array (nullable = false)
         |    |-- element: struct (containsNull = false)
         |    |    |-- name: string (nullable = false)
         |    |    |-- value: string (nullable = false)
        <BLANKLINE>
        >>> res.show(10, False)
        +---------------------------------------------------------------------------------+
        |values                                                                           |
        +---------------------------------------------------------------------------------+
        |[{pokemon.id, 4}, {pokemon.name, Charmander}, {pokemon.types, [{Fire}]}]         |
        |[{pokemon.id, 5}, {pokemon.name, Charmeleon}, {pokemon.types, [{Fire}]}]         |
        |[{pokemon.id, 6}, {pokemon.name, Charizard}, {pokemon.types, [{Fire}, {Flying}]}]|
        +---------------------------------------------------------------------------------+
        <BLANKLINE>
    """
    return f.array(
        *[
            f.struct(f.lit(c).alias(col_name_alias), f.col(quote(c)).astype(StringType()).alias(col_value_alias))
            for c in columns
        ]
    )


def nullable(col: Column) -> Column:
    """Make a `pyspark.sql.Column` nullable.
    This is especially useful for literal which are always non-nullable by default.

    Examples:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> from pyspark.sql import functions as f
        >>> df = spark.sql('''SELECT 1 as a''').withColumn("b", f.lit("2"))
        >>> df.printSchema()
        root
         |-- a: integer (nullable = false)
         |-- b: string (nullable = false)
        <BLANKLINE>
        >>> res = df.withColumn('a', nullable(f.col('a'))).withColumn('b', nullable(f.col('b')))
        >>> res.printSchema()
        root
         |-- a: integer (nullable = true)
         |-- b: string (nullable = true)
        <BLANKLINE>
    """
    return f.when(~col.isNull(), col)
