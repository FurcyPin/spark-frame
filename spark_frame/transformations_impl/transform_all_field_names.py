from typing import Callable

from pyspark.sql import DataFrame

from spark_frame.nested_impl.package import build_transformation_from_schema


def transform_all_field_names(df: DataFrame, transformation: Callable[[str], str]) -> DataFrame:
    """Apply a transformation to all nested field names of a DataFrame.

    !!! info
        This method is compatible with any schema. It recursively applies on structs, arrays and maps
        and is compatible with field names containing special characters.

    Args:
        df: A Spark DataFrame
        transformation: Transformation to apply to all field names in the DataFrame.

    Returns:
        A new DataFrame

    Examples:
        _**Example 1: with a nested schema structure**_

        In this example we cast all the field names of the schema to uppercase:
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

        >>> new_df = df.transform(transform_all_field_names, lambda s: s.upper())
        >>> nested.print_schema(new_df)
        root
         |-- NAME: string (nullable = false)
         |-- S1!.A: integer (nullable = false)
         |-- S2!!: integer (nullable = false)
         |-- S3!!.A: integer (nullable = false)
         |-- S4!.A!: integer (nullable = false)
         |-- S5!.A!.B.C: integer (nullable = false)
        <BLANKLINE>

        _**Example 2: sanitizing field names**_

        In this example we replace all dots and exclamation marks in field names with underscores.
        This is useful to make a DataFrame compatible with the [spark_frame.nested](/spark-frame/reference/nested)
        module.
        >>> df = spark.sql('''SELECT
        ...     ARRAY(STRUCT(
        ...         ARRAY(STRUCT(
        ...             STRUCT(1 as `d.d!d`) as `c.c!c`
        ...         )) as `b.b!b`
        ...    )) as `a.a!a`
        ... ''')
        >>> df.printSchema()
        root
         |-- a.a!a: array (nullable = false)
         |    |-- element: struct (containsNull = false)
         |    |    |-- b.b!b: array (nullable = false)
         |    |    |    |-- element: struct (containsNull = false)
         |    |    |    |    |-- c.c!c: struct (nullable = false)
         |    |    |    |    |    |-- d.d!d: integer (nullable = false)
        <BLANKLINE>
        >>> new_df = df.transform(transform_all_field_names, lambda s: s.replace(".","_").replace("!", "_"))
        >>> new_df.printSchema()
        root
         |-- a_a_a: array (nullable = false)
         |    |-- element: struct (containsNull = false)
         |    |    |-- b_b_b: array (nullable = false)
         |    |    |    |-- element: struct (containsNull = false)
         |    |    |    |    |-- c_c_c: struct (nullable = false)
         |    |    |    |    |    |-- d_d_d: integer (nullable = false)
        <BLANKLINE>

        This also works on fields of type `MAP<K,V>`.
        >>> df = spark.sql('SELECT MAP(STRUCT(1 as `a.a!a`), STRUCT(2 as `b.b!b`)) as `m.m!m`')
        >>> df.printSchema()
        root
         |-- m.m!m: map (nullable = false)
         |    |-- key: struct
         |    |    |-- a.a!a: integer (nullable = false)
         |    |-- value: struct (valueContainsNull = false)
         |    |    |-- b.b!b: integer (nullable = false)
        <BLANKLINE>
        >>> new_df = df.transform(transform_all_field_names, lambda s: s.replace(".","_").replace("!", "_"))
        >>> new_df.printSchema()
        root
         |-- m_m_m: map (nullable = false)
         |    |-- key: struct
         |    |    |-- a_a_a: integer (nullable = false)
         |    |-- value: struct (valueContainsNull = false)
         |    |    |-- b_b_b: integer (nullable = false)
        <BLANKLINE>
    """
    root_transformation = build_transformation_from_schema(df.schema, name_transformation=transformation)
    return df.select(*root_transformation(df))
