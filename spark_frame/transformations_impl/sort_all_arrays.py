from typing import Optional

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import ArrayType, DataType

from spark_frame.transformations_impl.transform_all_fields import transform_all_fields


def sort_all_arrays(df: DataFrame) -> DataFrame:
    """Given a DataFrame, sort all fields of type `ARRAY` in a canonical order, making them comparable.
    This also applies to nested fields, even those inside other arrays.

    !!! warning "Limitation"
        - Arrays containing sub-fields of type Map cannot be sorted, as the Map type is not comparable.
        - A possible workaround is to first use the transformation
        [`spark_frame.transformations.convert_all_maps_to_arrays`]
        [spark_frame.transformations_impl.convert_all_maps_to_arrays.convert_all_maps_to_arrays]

    Args:
        df: A Spark DataFrame

    Returns:
        A new DataFrame where all arrays have been sorted.

    Examples:
        *Example 1:* with a simple `ARRAY<INT>`

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('SELECT 1 as id, ARRAY(3, 2, 1) as a')
        >>> df.show()
        +---+---------+
        | id|        a|
        +---+---------+
        |  1|[3, 2, 1]|
        +---+---------+
        <BLANKLINE>
        >>> sort_all_arrays(df).show()
        +---+---------+
        | id|        a|
        +---+---------+
        |  1|[1, 2, 3]|
        +---+---------+
        <BLANKLINE>

        *Example 2:* with an `ARRAY<STRUCT<...>>`

        >>> df = spark.sql('SELECT ARRAY(STRUCT(2 as a, 1 as b), STRUCT(1 as a, 2 as b), STRUCT(1 as a, 1 as b)) as s')
        >>> df.show(truncate=False)
        +------------------------+
        |s                       |
        +------------------------+
        |[{2, 1}, {1, 2}, {1, 1}]|
        +------------------------+
        <BLANKLINE>
        >>> df.transform(sort_all_arrays).show(truncate=False)
        +------------------------+
        |s                       |
        +------------------------+
        |[{1, 1}, {1, 2}, {2, 1}]|
        +------------------------+
        <BLANKLINE>

        *Example 3:* with an `ARRAY<STRUCT<STRUCT<...>>>`

        >>> df = spark.sql('''SELECT ARRAY(
        ...         STRUCT(STRUCT(2 as a, 2 as b) as s),
        ...         STRUCT(STRUCT(1 as a, 2 as b) as s)
        ...     ) as l1
        ... ''')
        >>> df.show(truncate=False)
        +--------------------+
        |l1                  |
        +--------------------+
        |[{{2, 2}}, {{1, 2}}]|
        +--------------------+
        <BLANKLINE>
        >>> df.transform(sort_all_arrays).show(truncate=False)
        +--------------------+
        |l1                  |
        +--------------------+
        |[{{1, 2}}, {{2, 2}}]|
        +--------------------+
        <BLANKLINE>

        *Example 4:* with an `ARRAY<ARRAY<ARRAY<INT>>>`

        As this example shows, the innermost arrays are sorted before the outermost arrays.

        >>> df = spark.sql('''SELECT ARRAY(
        ...         ARRAY(ARRAY(4, 1), ARRAY(3, 2)),
        ...         ARRAY(ARRAY(2, 2), ARRAY(2, 1))
        ...     ) as l1
        ... ''')
        >>> df.show(truncate=False)
        +------------------------------------+
        |l1                                  |
        +------------------------------------+
        |[[[4, 1], [3, 2]], [[2, 2], [2, 1]]]|
        +------------------------------------+
        <BLANKLINE>
        >>> df.transform(sort_all_arrays).show(truncate=False)
        +------------------------------------+
        |l1                                  |
        +------------------------------------+
        |[[[1, 2], [2, 2]], [[1, 4], [2, 3]]]|
        +------------------------------------+
        <BLANKLINE>
    """

    def sort_array(col: Column, data_type: DataType) -> Optional[Column]:
        if isinstance(data_type, ArrayType):
            return f.sort_array(col)
        else:
            return None

    return transform_all_fields(df, sort_array)
