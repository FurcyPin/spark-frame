from typing import List

from pyspark.sql import DataFrame

from spark_frame.data_type_utils import flatten_schema


def fields(df: DataFrame) -> List[str]:
    """Return the name of all the fields (including nested sub-fields) in the given DataFrame.

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

    Returns:
        The list of all flattened field names in this DataFrame

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql import functions as f
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''SELECT
        ...     1 as id,
        ...     ARRAY(STRUCT(2 as a, ARRAY(STRUCT(3 as c, 4 as d)) as b, ARRAY(5, 6) as e)) as s1,
        ...     STRUCT(7 as f) as s2,
        ...     ARRAY(ARRAY(1, 2), ARRAY(3, 4)) as s3,
        ...     ARRAY(ARRAY(STRUCT(1 as c, 2 as d)), ARRAY(STRUCT(3 as c, 4 as d))) as s4
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
         |    |    |    |-- c: integer (nullable = false)
         |    |    |    |-- d: integer (nullable = false)
        <BLANKLINE>
        >>> for field in fields(df): print(field)
        id
        s1!.a
        s1!.b!.c
        s1!.b!.d
        s1!.e!
        s2.f
        s3!!
        s4!!.c
        s4!!.d
    """
    return [col.name for col in flatten_schema(df.schema, explode=True)]
