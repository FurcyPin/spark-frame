from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import ArrayType, DataType

from spark_frame.transformations_impl.transform_all_fields import transform_all_fields


def flatten_all_arrays(df: DataFrame) -> DataFrame:
    """Flatten all columns of type `ARRAY<ARRAY<T>>` inside the given DataFrame into `ARRAY<<T>>>`.
    This transformation works recursively on every nesting level.

    !!! info
        This method is compatible with any schema. It recursively applies on structs, arrays and maps
        and accepts field names containing dots (`.`), exclamation marks (`!`) or percentage (`%`).

    Args:
        df: A Spark DataFrame

    Returns:
        A new DataFrame in which all arrays of array have been flattened

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('SELECT 1 as id, ARRAY(ARRAY(ARRAY(1, 2), ARRAY(3)), ARRAY(ARRAY(4), ARRAY(5))) as a')
        >>> df.printSchema()
        root
         |-- id: integer (nullable = false)
         |-- a: array (nullable = false)
         |    |-- element: array (containsNull = false)
         |    |    |-- element: array (containsNull = false)
         |    |    |    |-- element: integer (containsNull = false)
        <BLANKLINE>
        >>> df.show(truncate=False)
        +---+---------------------------+
        |id |a                          |
        +---+---------------------------+
        |1  |[[[1, 2], [3]], [[4], [5]]]|
        +---+---------------------------+
        <BLANKLINE>
        >>> res_df = flatten_all_arrays(df)
        >>> res_df.printSchema()
        root
         |-- id: integer (nullable = false)
         |-- a: array (nullable = false)
         |    |-- element: integer (containsNull = false)
        <BLANKLINE>
        >>> res_df.show(truncate=False)
        +---+---------------+
        |id |a              |
        +---+---------------+
        |1  |[1, 2, 3, 4, 5]|
        +---+---------------+
        <BLANKLINE>
    """

    def flatten_array(col: Column, data_type: DataType):
        if isinstance(data_type, ArrayType) and isinstance(data_type.elementType, ArrayType):
            return f.flatten(col)

    return transform_all_fields(df, flatten_array)
