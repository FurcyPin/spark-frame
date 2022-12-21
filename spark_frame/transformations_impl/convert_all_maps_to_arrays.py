from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import MapType, StructField

from spark_frame import fp
from spark_frame.conf import REPETITION_MARKER, STRUCT_SEPARATOR
from spark_frame.data_type_utils import flatten_schema
from spark_frame.fp import PrintableFunction, higher_order
from spark_frame.nested_impl.package import _deepest_granularity, resolve_nested_fields


def convert_all_maps_to_arrays(df: DataFrame) -> DataFrame:
    """Transform all columns of type `Map<K,V>` inside the given DataFrame into `ARRAY<STRUCT<key: K, value: V>>`.
    This transformation works recursively on every nesting level.

    !!! warning "Limitations"
        Currently, this method does not work on DataFrames with field names containing dots (`.`)
        or exclamation marks (`!`).

    Args:
        df: A Spark DataFrame

    Returns:
        A new DataFrame in which all maps have been replaced with arrays of entries.

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('SELECT 1 as id, ARRAY(MAP(1, STRUCT(MAP(1, "a") as m2))) as m1')
        >>> df.printSchema()
        root
         |-- id: integer (nullable = false)
         |-- m1: array (nullable = false)
         |    |-- element: map (containsNull = false)
         |    |    |-- key: integer
         |    |    |-- value: struct (valueContainsNull = false)
         |    |    |    |-- m2: map (nullable = false)
         |    |    |    |    |-- key: integer
         |    |    |    |    |-- value: string (valueContainsNull = false)
        <BLANKLINE>
        >>> df.show()
        +---+-------------------+
        | id|                 m1|
        +---+-------------------+
        |  1|[{1 -> {{1 -> a}}}]|
        +---+-------------------+
        <BLANKLINE>
        >>> res_df = convert_all_maps_to_arrays(df)
        >>> res_df.printSchema()
        root
         |-- id: integer (nullable = false)
         |-- m1: array (nullable = false)
         |    |-- element: array (containsNull = false)
         |    |    |-- element: struct (containsNull = false)
         |    |    |    |-- key: integer (nullable = false)
         |    |    |    |-- value: struct (nullable = false)
         |    |    |    |    |-- m2: array (nullable = false)
         |    |    |    |    |    |-- element: struct (containsNull = false)
         |    |    |    |    |    |    |-- key: integer (nullable = false)
         |    |    |    |    |    |    |-- value: string (nullable = false)
        <BLANKLINE>
        >>> res_df.show()
        +---+-------------------+
        | id|                 m1|
        +---+-------------------+
        |  1|[[{1, {[{1, a}]}}]]|
        +---+-------------------+
        <BLANKLINE>
    """

    def build_col(field: StructField) -> PrintableFunction:
        parent_structs = _deepest_granularity(field.name)
        if isinstance(field.dataType, MapType):
            f1 = PrintableFunction(lambda s: f.map_entries(s), lambda s: f"f.map_entries({s})")
        else:
            f1 = higher_order.identity
        f2 = higher_order.recursive_struct_get(parent_structs)
        return fp.compose(f1, f2)

    do_continue = True
    res_df = df
    while do_continue:
        schema_flat = flatten_schema(
            res_df.schema, explode=True, struct_separator=STRUCT_SEPARATOR, repetition_marker=REPETITION_MARKER
        )
        schema_contains_map = any([isinstance(field.dataType, MapType) for field in schema_flat])
        do_continue = schema_contains_map
        fields = {field.name: build_col(field) for field in schema_flat}
        res_df = res_df.select(*resolve_nested_fields(fields, starting_level=res_df))
    return res_df
