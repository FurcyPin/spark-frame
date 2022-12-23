from pyspark.sql import DataFrame

from spark_frame.nested_impl.schema_string import schema_string


def print_schema(df: DataFrame) -> None:
    """Print the DataFrame's flattened schema to the standard output.

    - Structs are flattened with a `.` after their name.
    - Arrays are flattened with a `!` character after their name.
    - Maps are flattened with a `%key` and '%value' after their name.

    !!! warning "Limitation: Dots, percents, and exclamation marks are not supported in field names"
        Given the syntax used, every method defined in the `spark_frame.nested` module assumes that all field
        names in DataFrames do not contain any dot `.`, percent `%` or exclamation mark `!`.
        This can be worked around using the transformation
        [`spark_frame.transformations.transform_all_field_names`]
        [spark_frame.transformations_impl.transform_all_field_names.transform_all_field_names].

    Args:
        df: A Spark DataFrame

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> from spark_frame import nested
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''SELECT
        ...     1 as id,
        ...     ARRAY(STRUCT(2 as a, ARRAY(STRUCT(3 as c, 4 as d)) as b, ARRAY(5, 6) as e)) as s1,
        ...     STRUCT(7 as f) as s2,
        ...     ARRAY(ARRAY(1, 2), ARRAY(3, 4)) as s3,
        ...     ARRAY(ARRAY(STRUCT(1 as e, 2 as f)), ARRAY(STRUCT(3 as e, 4 as f))) as s4,
        ...     MAP(STRUCT(1 as a), STRUCT(2 as b)) as m1
        ... ''')
        >>> df.printSchema()
        root
         |-- id: integer (nullable = false)
         |-- s1: array (nullable = false)
         |    |-- element: struct (containsNull = false)
         |    |    |-- a: integer (nullable = false)
         |    |    |-- b: array (nullable = false)
         |    |    |    |-- element: struct (containsNull = false)
         |    |    |    |    |-- c: integer (nullable = false)
         |    |    |    |    |-- d: integer (nullable = false)
         |    |    |-- e: array (nullable = false)
         |    |    |    |-- element: integer (containsNull = false)
         |-- s2: struct (nullable = false)
         |    |-- f: integer (nullable = false)
         |-- s3: array (nullable = false)
         |    |-- element: array (containsNull = false)
         |    |    |-- element: integer (containsNull = false)
         |-- s4: array (nullable = false)
         |    |-- element: array (containsNull = false)
         |    |    |-- element: struct (containsNull = false)
         |    |    |    |-- e: integer (nullable = false)
         |    |    |    |-- f: integer (nullable = false)
         |-- m1: map (nullable = false)
         |    |-- key: struct
         |    |    |-- a: integer (nullable = false)
         |    |-- value: struct (valueContainsNull = false)
         |    |    |-- b: integer (nullable = false)
        <BLANKLINE>
        >>> nested.print_schema(df)
        root
         |-- id: integer (nullable = false)
         |-- s1!.a: integer (nullable = false)
         |-- s1!.b!.c: integer (nullable = false)
         |-- s1!.b!.d: integer (nullable = false)
         |-- s1!.e!: integer (nullable = false)
         |-- s2.f: integer (nullable = false)
         |-- s3!!: integer (nullable = false)
         |-- s4!!.e: integer (nullable = false)
         |-- s4!!.f: integer (nullable = false)
         |-- m1%key.a: integer (nullable = false)
         |-- m1%value.b: integer (nullable = false)
        <BLANKLINE>
    """
    print(schema_string(df))
