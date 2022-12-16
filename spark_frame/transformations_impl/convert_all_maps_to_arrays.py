from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import MapType, StructField

from spark_frame import fp
from spark_frame.conf import REPETITION_MARKER, STRUCT_SEPARATOR
from spark_frame.data_type_utils import flatten_schema
from spark_frame.fp import PrintableFunction, higher_order
from spark_frame.nested import resolve_nested_columns


def convert_all_maps_to_arrays(df: DataFrame) -> DataFrame:
    """Transform all columns of type `Map<K,V>` inside the given DataFrame into `ARRAY<STRUCT<key: K, value: V>>`.
    This transformation works recursively on every nesting level.

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
        is_repeated = field.name[-1] == REPETITION_MARKER
        col = field.name.split(STRUCT_SEPARATOR)[-1]
        if isinstance(field.dataType, MapType):
            f1 = PrintableFunction(lambda s: f.map_entries(s), lambda s: f"f.map_entries({s})")
        else:
            f1 = higher_order.identity
        if is_repeated:
            f2 = higher_order.identity
        else:
            f2 = higher_order.safe_struct_get(col)
        return fp.compose(f1, f2)

    do_continue = True
    while do_continue:
        schema_flat = flatten_schema(
            df.schema, explode=True, struct_separator=STRUCT_SEPARATOR, repetition_marker=REPETITION_MARKER
        )
        schema_contains_map = any([isinstance(field.dataType, MapType) for field in schema_flat])
        do_continue = schema_contains_map
        columns = {field.name: build_col(field) for field in schema_flat}
        df = df.select(*resolve_nested_columns(columns))
    return df
