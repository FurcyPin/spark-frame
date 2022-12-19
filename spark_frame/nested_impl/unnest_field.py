from typing import List, Optional

from pyspark.sql import DataFrame

from spark_frame.nested_impl.package import unnest_fields


def unnest_field(df: DataFrame, field_name: str, keep_columns: Optional[List[str]] = None) -> DataFrame:
    """Given a DataFrame, return a new DataFrame where the specified column has been recursively
    unnested (a.k.a. exploded).

    Args:
        df: A Spark DataFrame
        field_name: The name of a nested column to unnest
        keep_columns: List of column names to keep while unnesting

    Returns:
        A new DataFrame

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql import functions as f
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''SELECT
        ...     1 as id,
        ...     ARRAY(ARRAY(1, 2), ARRAY(3, 4)) as arr
        ... ''')
        >>> df.show(truncate=False)
        +---+----------------+
        |id |arr             |
        +---+----------------+
        |1  |[[1, 2], [3, 4]]|
        +---+----------------+
        <BLANKLINE>
        >>> from spark_frame import nested
        >>> nested.fields(df)
        ['id', 'arr!!']
        >>> unnest_field(df, 'arr!').show(truncate=False)
        +------+
        |arr!  |
        +------+
        |[1, 2]|
        |[3, 4]|
        +------+
        <BLANKLINE>
        >>> unnest_field(df, 'arr!!').show(truncate=False)
        +-----+
        |arr!!|
        +-----+
        |1    |
        |2    |
        |3    |
        |4    |
        +-----+
        <BLANKLINE>
        >>> unnest_field(df, 'arr!!', keep_columns=["id"]).show(truncate=False)
        +---+-----+
        |id |arr!!|
        +---+-----+
        |1  |1    |
        |1  |2    |
        |1  |3    |
        |1  |4    |
        +---+-----+
        <BLANKLINE>

        >>> df = spark.sql('''SELECT
        ...     1 as id,
        ...     ARRAY(
        ...         STRUCT(ARRAY(STRUCT("a1" as a, "b1" as b), STRUCT("a2" as a, "b1" as b)) as s2),
        ...         STRUCT(ARRAY(STRUCT("a3" as a, "b3" as b)) as s2)
        ...     ) as s1
        ... ''')
        >>> df.show(truncate=False)
        +---+--------------------------------------+
        |id |s1                                    |
        +---+--------------------------------------+
        |1  |[{[{a1, b1}, {a2, b1}]}, {[{a3, b3}]}]|
        +---+--------------------------------------+
        <BLANKLINE>
        >>> nested.fields(df)
        ['id', 's1!.s2!.a', 's1!.s2!.b']
        >>> unnest_field(df, 's1!.s2!').show(truncate=False)
        +--------+
        |s1!.s2! |
        +--------+
        |{a1, b1}|
        |{a2, b1}|
        |{a3, b3}|
        +--------+
        <BLANKLINE>

    """
    if keep_columns is None:
        keep_columns = []
    return unnest_fields(df, field_name, keep_columns=keep_columns)[0]
