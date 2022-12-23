from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.types import StructField

from spark_frame.data_type_utils import flatten_schema


def _flat_schema_to_tree_string(schema: List[StructField]) -> str:
    """Generates a string representing a flat schema in tree format"""

    def str_gen_schema_field(struct_field: StructField, prefix: str) -> List[str]:
        res = [
            f"{prefix}{struct_field.name}: {struct_field.dataType.typeName()} "
            f"(nullable = {str(struct_field.nullable).lower()})"
        ]
        return res

    def str_gen_schema(schema: List[StructField], prefix: str) -> List[str]:
        return [str for schema_field in schema for str in str_gen_schema_field(schema_field, prefix)]

    res = ["root"] + str_gen_schema(schema, " |-- ")

    return "\n".join(res) + "\n"


def schema_string(df: DataFrame) -> str:
    """Write the DataFrame's flattened schema to a string.

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
        a string representing the flattened schema

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> from spark_frame import nested
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''SELECT
        ...     1 as id,
        ...     ARRAY(STRUCT(2 as a, ARRAY(STRUCT(3 as c, 4 as d)) as b, ARRAY(5, 6) as e)) as s1,
        ...     STRUCT(7 as f) as s2,
        ...     ARRAY(ARRAY(1, 2), ARRAY(3, 4)) as s3,
        ...     ARRAY(ARRAY(STRUCT(1 as e, 2 as f)), ARRAY(STRUCT(3 as e, 4 as f))) as s4
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
        <BLANKLINE>
        >>> print(nested.schema_string(df))
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
        <BLANKLINE>
    """
    flat_schema = flatten_schema(df.schema, explode=True)
    return _flat_schema_to_tree_string(flat_schema.fields)
