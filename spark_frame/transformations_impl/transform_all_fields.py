from typing import Callable

from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DataType

from spark_frame.nested_impl.package import build_transformation_from_schema


def transform_all_fields(df: DataFrame, transformation: Callable[[Column, DataType], Column]) -> DataFrame:
    """Apply a transformation to all nested fields of a DataFrame.

    !!! info
        This method is compatible with any schema. It recursively applies on structs, arrays and maps
        and is compatible with field names containing special characters.

    Args:
        df: A Spark DataFrame
        transformation: Transformation to apply to all fields of the DataFrame. The transformation must take as input
            a Column expression and the DataType of the corresponding expression.

    Returns:
        A new DataFrame

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> from spark_frame import nested
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''SELECT
        ...     "John" as name,
        ...     ARRAY(STRUCT(1 as a), STRUCT(2 as a)) as s1,
        ...     ARRAY(ARRAY(1, 2), ARRAY(3, 4)) as s2,
        ...     ARRAY(ARRAY(STRUCT(1 as a)), ARRAY(STRUCT(2 as a))) as s3,
        ...     ARRAY(STRUCT(ARRAY(1, 2) as a), STRUCT(ARRAY(3, 4) as a)) as s4,
        ...     ARRAY(
        ...         STRUCT(ARRAY(STRUCT(STRUCT(1 as c) as b), STRUCT(STRUCT(2 as c) as b)) as a),
        ...         STRUCT(ARRAY(STRUCT(STRUCT(3 as c) as b), STRUCT(STRUCT(4 as c) as b)) as a)
        ...     ) as s5
        ... ''')
        >>> nested.print_schema(df)
        root
         |-- name: string (nullable = false)
         |-- s1!.a: integer (nullable = false)
         |-- s2!!: integer (nullable = false)
         |-- s3!!.a: integer (nullable = false)
         |-- s4!.a!: integer (nullable = false)
         |-- s5!.a!.b.c: integer (nullable = false)
        <BLANKLINE>
        >>> df.show(truncate=False)
        +----+----------+----------------+--------------+--------------------+------------------------------------+
        |name|s1        |s2              |s3            |s4                  |s5                                  |
        +----+----------+----------------+--------------+--------------------+------------------------------------+
        |John|[{1}, {2}]|[[1, 2], [3, 4]]|[[{1}], [{2}]]|[{[1, 2]}, {[3, 4]}]|[{[{{1}}, {{2}}]}, {[{{3}}, {{4}}]}]|
        +----+----------+----------------+--------------+--------------------+------------------------------------+
        <BLANKLINE>
        >>> from pyspark.sql.types import IntegerType
        >>> def cast_int_as_double(col: Column, data_type: DataType):
        ...     if isinstance(data_type, IntegerType):
        ...         return col.cast("DOUBLE")
        >>> new_df = df.transform(transform_all_fields, cast_int_as_double)
        >>> nested.print_schema(new_df)
        root
         |-- name: string (nullable = false)
         |-- s1!.a: double (nullable = false)
         |-- s2!!: double (nullable = false)
         |-- s3!!.a: double (nullable = false)
         |-- s4!.a!: double (nullable = false)
         |-- s5!.a!.b.c: double (nullable = false)
        <BLANKLINE>
        >>> new_df.show(truncate=False)  # noqa: E501
        +----+--------------+------------------------+------------------+----------------------------+--------------------------------------------+
        |name|s1            |s2                      |s3                |s4                          |s5                                          |
        +----+--------------+------------------------+------------------+----------------------------+--------------------------------------------+
        |John|[{1.0}, {2.0}]|[[1.0, 2.0], [3.0, 4.0]]|[[{1.0}], [{2.0}]]|[{[1.0, 2.0]}, {[3.0, 4.0]}]|[{[{{1.0}}, {{2.0}}]}, {[{{3.0}}, {{4.0}}]}]|
        +----+--------------+------------------------+------------------+----------------------------+--------------------------------------------+
        <BLANKLINE>
    """
    root_transformation = build_transformation_from_schema(df.schema, column_transformation=transformation)
    return df.select(*root_transformation(df))
