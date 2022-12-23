from typing import List, Optional

from pyspark.sql import DataFrame

from spark_frame import nested
from spark_frame.nested_impl.package import unnest_fields
from spark_frame.utils import is_sub_field_of_any


def unnest_all_fields(df: DataFrame, keep_columns: Optional[List[str]] = None) -> List[DataFrame]:
    """Given a DataFrame, return a list of DataFrames where all arrays have been recursively unnested (a.k.a. exploded).
    This produce one DataFrame for each possible granularity.

    For instance, given a DataFrame with the following flattened schema:
        id
        s1.a
        s2!.b
        s2!.c
        s2!.s3!.d
        s4!.e
        s4!.f

    This will return four DataFrames containing the following unnested columns:
        - id, s1.a
        - s2!.b, s2!.c
        - s2!.s3!.d
        - s4!.e, s4!.f

    !!! warning "Limitation: Maps are not unnested"
        - Fields of type Maps are not unnested by this method.
        - A possible workaround is to first use the transformation
        [`spark_frame.transformations.convert_all_maps_to_arrays`]
        [spark_frame.transformations_impl.convert_all_maps_to_arrays.convert_all_maps_to_arrays]

    Args:
        df: A Spark DataFrame
        keep_columns: Names of columns that should be kept while unnesting

    Returns:
        A list of DataFrames

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> from spark_frame import nested
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''
        ...     SELECT
        ...         1 as id,
        ...         STRUCT(2 as a) as s1,
        ...         ARRAY(STRUCT(3 as b, 4 as c, ARRAY(STRUCT(5 as d), STRUCT(6 as d)) as s3)) as s2,
        ...         ARRAY(STRUCT(7 as e, 8 as f), STRUCT(9 as e, 10 as f)) as s4
        ... ''')
        >>> df.show(truncate=False)
        +---+---+--------------------+-----------------+
        |id |s1 |s2                  |s4               |
        +---+---+--------------------+-----------------+
        |1  |{2}|[{3, 4, [{5}, {6}]}]|[{7, 8}, {9, 10}]|
        +---+---+--------------------+-----------------+
        <BLANKLINE>
        >>> nested.fields(df)
        ['id', 's1.a', 's2!.b', 's2!.c', 's2!.s3!.d', 's4!.e', 's4!.f']
        >>> result_df_list = nested.unnest_all_fields(df, keep_columns=["id"])
        >>> for result_df in result_df_list: result_df.show()
        +---+----+
        | id|s1.a|
        +---+----+
        |  1|   2|
        +---+----+
        <BLANKLINE>
        +---+-----+-----+
        | id|s2!.b|s2!.c|
        +---+-----+-----+
        |  1|    3|    4|
        +---+-----+-----+
        <BLANKLINE>
        +---+---------+
        | id|s2!.s3!.d|
        +---+---------+
        |  1|        5|
        |  1|        6|
        +---+---------+
        <BLANKLINE>
        +---+-----+-----+
        | id|s4!.e|s4!.f|
        +---+-----+-----+
        |  1|    7|    8|
        |  1|    9|   10|
        +---+-----+-----+
        <BLANKLINE>
    """
    if keep_columns is None:
        keep_columns = []
    fields_to_unnest = [field for field in nested.fields(df) if not is_sub_field_of_any(field, keep_columns)]
    return unnest_fields(df, fields_to_unnest, keep_columns=keep_columns)
