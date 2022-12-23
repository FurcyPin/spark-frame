from pyspark.sql import DataFrame

from spark_frame.utils import assert_true


def union_dataframes(*dfs: DataFrame) -> DataFrame:
    """Returns the union between multiple DataFrames

    Args:
        dfs: One or more Spark DataFrames

    Returns:
        A new DataFrame containing the union of all input DataFrames

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df1 = spark.sql('SELECT 1 as a')
        >>> df2 = spark.sql('SELECT 2 as a')
        >>> df3 = spark.sql('SELECT 3 as a')
        >>> union_dataframes(df1, df2, df3).show()
        +---+
        |  a|
        +---+
        |  1|
        |  2|
        |  3|
        +---+
        <BLANKLINE>
        >>> df1.transform(union_dataframes, df2, df3).show()
        +---+
        |  a|
        +---+
        |  1|
        |  2|
        |  3|
        +---+
        <BLANKLINE>
    """
    assert_true(len(dfs) > 0, ValueError("Input list is empty"))
    res = dfs[0]
    for df in dfs[1:]:
        res = res.union(df)
    return res
