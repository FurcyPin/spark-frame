from typing import Optional

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import DataType, MapType

from spark_frame.transformations_impl.transform_all_fields import transform_all_fields


def convert_all_maps_to_arrays(df: DataFrame) -> DataFrame:
    """Transform all columns of type `Map<K,V>` inside the given DataFrame into `ARRAY<STRUCT<key: K, value: V>>`.
    This transformation works recursively on every nesting level.

    !!! info
        This method is compatible with any schema. It recursively applies on structs, arrays and maps
        and is compatible with field names containing special characters.

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

    def map_to_arrays(col: Column, data_type: DataType) -> Optional[Column]:
        if isinstance(data_type, MapType):
            return f.map_entries(col)
        else:
            return None

    return transform_all_fields(df, map_to_arrays)
