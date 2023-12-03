from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from spark_frame import nested
from spark_frame.utils import schema_string, strip_margin


def test_with_fields(spark: SparkSession):
    """
    GIVEN a DataFrame with nested fields
    WHEN we use with_fields to add a new field
    THEN the the other fields should remain undisturbed
    """
    df = spark.sql(
        """SELECT
        1 as id,
        ARRAY(STRUCT(2 as a, ARRAY(STRUCT(3 as c, 4 as d)) as b, ARRAY(5, 6) as e)) as s1,
        STRUCT(7 as f) as s2,
        ARRAY(ARRAY(1, 2), ARRAY(3, 4)) as s3,
        ARRAY(ARRAY(STRUCT(1 as e, 2 as f)), ARRAY(STRUCT(3 as e, 4 as f))) as s4
    """,
    )
    assert schema_string(df) == strip_margin(
        """
        |root
        | |-- id: integer (nullable = false)
        | |-- s1: array (nullable = false)
        | |    |-- element: struct (containsNull = false)
        | |    |    |-- a: integer (nullable = false)
        | |    |    |-- b: array (nullable = false)
        | |    |    |    |-- element: struct (containsNull = false)
        | |    |    |    |    |-- c: integer (nullable = false)
        | |    |    |    |    |-- d: integer (nullable = false)
        | |    |    |-- e: array (nullable = false)
        | |    |    |    |-- element: integer (containsNull = false)
        | |-- s2: struct (nullable = false)
        | |    |-- f: integer (nullable = false)
        | |-- s3: array (nullable = false)
        | |    |-- element: array (containsNull = false)
        | |    |    |-- element: integer (containsNull = false)
        | |-- s4: array (nullable = false)
        | |    |-- element: array (containsNull = false)
        | |    |    |-- element: struct (containsNull = false)
        | |    |    |    |-- e: integer (nullable = false)
        | |    |    |    |-- f: integer (nullable = false)
        |""",
    )
    new_df = df.transform(nested.with_fields, {"s5.g": f.col("s2.f").cast("DOUBLE")})
    assert schema_string(new_df) == strip_margin(
        """
        |root
        | |-- id: integer (nullable = false)
        | |-- s1: array (nullable = false)
        | |    |-- element: struct (containsNull = false)
        | |    |    |-- a: integer (nullable = false)
        | |    |    |-- b: array (nullable = false)
        | |    |    |    |-- element: struct (containsNull = false)
        | |    |    |    |    |-- c: integer (nullable = false)
        | |    |    |    |    |-- d: integer (nullable = false)
        | |    |    |-- e: array (nullable = false)
        | |    |    |    |-- element: integer (containsNull = false)
        | |-- s2: struct (nullable = false)
        | |    |-- f: integer (nullable = false)
        | |-- s3: array (nullable = false)
        | |    |-- element: array (containsNull = false)
        | |    |    |-- element: integer (containsNull = false)
        | |-- s4: array (nullable = false)
        | |    |-- element: array (containsNull = false)
        | |    |    |-- element: struct (containsNull = false)
        | |    |    |    |-- e: integer (nullable = false)
        | |    |    |    |-- f: integer (nullable = false)
        | |-- s5: struct (nullable = false)
        | |    |-- g: double (nullable = false)
        |""",
    )
