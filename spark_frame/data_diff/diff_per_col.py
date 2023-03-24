from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from spark_frame import nested
from spark_frame.data_diff.diff_results import DiffResult
from spark_frame.data_diff.diff_stats import DiffStats
from spark_frame.data_diff.package import STRUCT_SEPARATOR_REPLACEMENT


def _get_col_df(columns: List[str], spark: SparkSession) -> DataFrame:
    """Create a DataFrame listing the column names with their column number

    Examples:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = _get_col_df(["id", "c1", "c2"], spark)
        >>> df.printSchema()
        root
         |-- column_number: integer (nullable = true)
         |-- column_name: string (nullable = true)
        <BLANKLINE>
        >>> df.show()
        +-------------+-----------+
        |column_number|column_name|
        +-------------+-----------+
        |            0|         id|
        |            1|         c1|
        |            2|         c2|
        +-------------+-----------+
        <BLANKLINE>
    """
    col_df = spark.createDataFrame(list(enumerate(columns)), "column_number INT, column_name STRING").withColumn(
        "column_name", f.regexp_replace(f.col("column_name"), STRUCT_SEPARATOR_REPLACEMENT, ".")
    )
    return col_df


def _get_pivoted_df(top_per_col_state_df: DataFrame, max_nb_rows_per_col_state: int) -> DataFrame:
    """Pivot the top_per_col_state_df

    Examples:

        >>> from spark_frame.data_diff.diff_results import _get_test_diff_result
        >>> diff_result = _get_test_diff_result()
        >>> diff_result.top_per_col_state_df.show()
        +-----------+-------------+----------+-----------+---+---------------+-------+
        |column_name|        state|left_value|right_value| nb|col_state_total|row_num|
        +-----------+-------------+----------+-----------+---+---------------+-------+
        |         c1|    no_change|         b|          b|  3|              4|      1|
        |         c1|    no_change|         a|          a|  1|              4|      2|
        |         c1| only_in_left|         c|       null|  1|              1|      1|
        |         c1|only_in_right|      null|          f|  1|              1|      1|
        |         c2|      changed|         2|          4|  2|              3|      1|
        |         c2|      changed|         2|          3|  1|              3|      2|
        |         c2|    no_change|         1|          1|  1|              1|      1|
        |         c2| only_in_left|         3|       null|  1|              1|      1|
        |         c2|only_in_right|      null|          3|  1|              1|      1|
        |         id|    no_change|         1|          1|  1|              4|      1|
        |         id|    no_change|         2|          2|  1|              4|      2|
        |         id|    no_change|         3|          3|  1|              4|      3|
        |         id|    no_change|         4|          4|  1|              4|      4|
        |         id| only_in_left|         5|       null|  1|              1|      1|
        |         id|only_in_right|      null|          6|  1|              1|      1|
        +-----------+-------------+----------+-----------+---+---------------+-------+
        <BLANKLINE>

        >>> _get_pivoted_df(diff_result.top_per_col_state_df, max_nb_rows_per_col_state=10).show(truncate=False)  # noqa: E501
        +-----------+----------+----------------------+------------+--------------------------------------------+---------------+-----------------+----------------+------------------+
        |column_name|changed_nb|changed_diff          |no_change_nb|no_change_diff                              |only_in_left_nb|only_in_left_diff|only_in_right_nb|only_in_right_diff|
        +-----------+----------+----------------------+------------+--------------------------------------------+---------------+-----------------+----------------+------------------+
        |c1         |null      |[]                    |4           |[{b, b, 3}, {a, a, 1}]                      |1              |[{c, null, 1}]   |1               |[{null, f, 1}]    |
        |c2         |3         |[{2, 4, 2}, {2, 3, 1}]|1           |[{1, 1, 1}]                                 |1              |[{3, null, 1}]   |1               |[{null, 3, 1}]    |
        |id         |null      |[]                    |4           |[{1, 1, 1}, {2, 2, 1}, {3, 3, 1}, {4, 4, 1}]|1              |[{5, null, 1}]   |1               |[{null, 6, 1}]    |
        +-----------+----------+----------------------+------------+--------------------------------------------+---------------+-----------------+----------------+------------------+
        <BLANKLINE>
    """
    pivoted_df = (
        top_per_col_state_df.groupBy("column_name")
        .pivot("state", values=["changed", "no_change", "only_in_left", "only_in_right"])
        .agg(
            f.sum("nb").alias("nb"),
            f.expr(
                f"""
            ARRAY_AGG(
                CASE WHEN row_num <= {max_nb_rows_per_col_state} THEN STRUCT(left_value, right_value, nb) END
            )"""
            ).alias("diff"),
        )
    )
    return pivoted_df


def _format_diff_per_col_df(pivoted_df: DataFrame, col_df: DataFrame, diff_stats: DiffStats) -> DataFrame:
    """

    Examples

        >>> from spark_frame.data_diff.diff_results import _get_test_diff_result
        >>> diff_result = _get_test_diff_result()
        >>> spark = diff_result.top_per_col_state_df.sparkSession
        >>> columns = diff_result.diff_df.columns[:-2]
        >>> col_df = _get_col_df(columns, spark)
        >>> pivoted_df = _get_pivoted_df(diff_result.top_per_col_state_df, max_nb_rows_per_col_state=10)
        >>> col_df.show()
        +-------------+-----------+
        |column_number|column_name|
        +-------------+-----------+
        |            0|         id|
        |            1|         c1|
        |            2|         c2|
        +-------------+-----------+
        <BLANKLINE>
        >>> pivoted_df.show()  # noqa: E501
        +-----------+----------+--------------------+------------+--------------------+---------------+-----------------+----------------+------------------+
        |column_name|changed_nb|        changed_diff|no_change_nb|      no_change_diff|only_in_left_nb|only_in_left_diff|only_in_right_nb|only_in_right_diff|
        +-----------+----------+--------------------+------------+--------------------+---------------+-----------------+----------------+------------------+
        |         c1|      null|                  []|           4|[{b, b, 3}, {a, a...|              1|   [{c, null, 1}]|               1|    [{null, f, 1}]|
        |         c2|         3|[{2, 4, 2}, {2, 3...|           1|         [{1, 1, 1}]|              1|   [{3, null, 1}]|               1|    [{null, 3, 1}]|
        |         id|      null|                  []|           4|[{1, 1, 1}, {2, 2...|              1|   [{5, null, 1}]|               1|    [{null, 6, 1}]|
        +-----------+----------+--------------------+------------+--------------------+---------------+-----------------+----------------+------------------+
        <BLANKLINE>
        >>> diff_per_col_df = _format_diff_per_col_df(pivoted_df, col_df, diff_result.diff_stats)
        >>> nested.print_schema(diff_per_col_df)
        root
         |-- column_number: integer (nullable = true)
         |-- column_name: string (nullable = true)
         |-- counts.total: integer (nullable = false)
         |-- counts.changed: long (nullable = false)
         |-- counts.no_change: long (nullable = false)
         |-- counts.only_in_left: long (nullable = false)
         |-- counts.only_in_right: long (nullable = false)
         |-- diff.changed!.left_value: string (nullable = true)
         |-- diff.changed!.right_value: string (nullable = true)
         |-- diff.changed!.nb: long (nullable = false)
         |-- diff.no_change!.value: string (nullable = true)
         |-- diff.no_change!.nb: long (nullable = false)
         |-- diff.only_in_left!.value: string (nullable = true)
         |-- diff.only_in_left!.nb: long (nullable = false)
         |-- diff.only_in_right!.value: string (nullable = true)
         |-- diff.only_in_right!.nb: long (nullable = false)
        <BLANKLINE>
        >>> diff_per_col_df.show(truncate=False)
        +-------------+-----------+---------------+----------------------------------------------------------+
        |column_number|column_name|counts         |diff                                                      |
        +-------------+-----------+---------------+----------------------------------------------------------+
        |0            |id         |{6, 0, 4, 1, 1}|{[], [{1, 1}, {2, 1}, {3, 1}, {4, 1}], [{5, 1}], [{6, 1}]}|
        |1            |c1         |{6, 0, 4, 1, 1}|{[], [{b, 3}, {a, 1}], [{c, 1}], [{f, 1}]}                |
        |2            |c2         |{6, 3, 1, 1, 1}|{[{2, 4, 2}, {2, 3, 1}], [{1, 1}], [{3, 1}], [{3, 1}]}    |
        +-------------+-----------+---------------+----------------------------------------------------------+
        <BLANKLINE>

    """
    coalesced_df = (
        col_df.join(pivoted_df, "column_name", "left")
        .withColumn("changed_nb", f.coalesce(f.col("changed_nb"), f.lit(0)))
        .withColumn("no_change_nb", f.coalesce(f.col("no_change_nb"), f.lit(0)))
        .withColumn("only_in_left_nb", f.coalesce(f.col("only_in_left_nb"), f.lit(0)))
        .withColumn("only_in_right_nb", f.coalesce(f.col("only_in_right_nb"), f.lit(0)))
        .withColumn("changed_diff", f.coalesce(f.col("changed_diff"), f.array()))
        .withColumn("no_change_diff", f.coalesce(f.col("no_change_diff"), f.array()))
        .withColumn("only_in_left_diff", f.coalesce(f.col("only_in_left_diff"), f.array()))
        .withColumn("only_in_right_diff", f.coalesce(f.col("only_in_right_diff"), f.array()))
    )
    renamed_df = coalesced_df.select(
        f.col("column_number"),
        f.col("column_name"),
        f.struct(
            f.lit(diff_stats.total).alias("total"),
            f.col("changed_nb").alias("changed"),
            f.col("no_change_nb").alias("no_change"),
            f.col("only_in_left_nb").alias("only_in_left"),
            f.col("only_in_right_nb").alias("only_in_right"),
        ).alias("counts"),
        f.struct(
            f.col("changed_diff").alias("changed"),
            f.col("no_change_diff").alias("no_change"),
            f.col("only_in_left_diff").alias("only_in_left"),
            f.col("only_in_right_diff").alias("only_in_right"),
        ).alias("diff"),
    ).orderBy("column_number")
    formatted_df = renamed_df.transform(
        nested.select,
        {
            "column_number": None,
            "column_name": None,
            "counts": None,
            "diff.changed": None,
            "diff.no_change!.value": lambda s: s["left_value"],
            "diff.no_change!.nb": lambda s: s["nb"],
            "diff.only_in_left!.value": lambda s: s["left_value"],
            "diff.only_in_left!.nb": lambda s: s["nb"],
            "diff.only_in_right!.value": lambda s: s["right_value"],
            "diff.only_in_right!.nb": lambda s: s["nb"],
        },
    )
    return formatted_df


def get_diff_per_col_df(
    diff_result: DiffResult,
    max_nb_rows_per_col_state: int,
    top_per_col_state_df: Optional[DataFrame] = None,
) -> DataFrame:
    """Given a top_per_col_state_df, return a Dict[str, int] that gives for each column and each
    column state (changed, no_change, only_in_left, only_in_right) the total number of occurences
    and the most frequent occurrences.

    !!! warning
        The arrays contained in the field `diff` are NOT guaranteed to be sorted,
        and Spark currently does not provide any way to perform a sort_by on an ARRAY<STRUCT>.

    Args:
        diff_result: A DiffResult object
        max_nb_rows_per_col_state: The maximal size of the arrays in `diff`
        top_per_col_state_df: An optional alternative DataFrame, used only for testing.

    Returns:
        A DataFrame with the following schema:

            root
             |-- column_number: integer (nullable = true)
             |-- column_name: string (nullable = true)
             |-- counts.total: integer (nullable = false)
             |-- counts.changed: long (nullable = false)
             |-- counts.no_change: long (nullable = false)
             |-- counts.only_in_left: long (nullable = false)
             |-- counts.only_in_right: long (nullable = false)
             |-- diff.changed!.left_value: string (nullable = true)
             |-- diff.changed!.right_value: string (nullable = true)
             |-- diff.changed!.nb: long (nullable = false)
             |-- diff.no_change!.value: string (nullable = true)
             |-- diff.no_change!.nb: long (nullable = false)
             |-- diff.only_in_left!.value: string (nullable = true)
             |-- diff.only_in_left!.nb: long (nullable = false)
             |-- diff.only_in_right!.value: string (nullable = true)
             |-- diff.only_in_right!.nb: long (nullable = false)
            <BLANKLINE>

    Examples:
        >>> from spark_frame.data_diff.diff_results import _get_test_diff_result
        >>> diff_result = _get_test_diff_result()
        >>> diff_result.diff_df.show()
        +---+----------------+----------------+-------------+------------+
        | id|              c1|              c2|   __EXISTS__|__IS_EQUAL__|
        +---+----------------+----------------+-------------+------------+
        |  1|    {a, a, true}|    {1, 1, true}| {true, true}|        true|
        |  2|    {b, b, true}|   {2, 3, false}| {true, true}|       false|
        |  3|    {b, b, true}|   {2, 4, false}| {true, true}|       false|
        |  4|    {b, b, true}|   {2, 4, false}| {true, true}|       false|
        |  5|{c, null, false}|{3, null, false}|{true, false}|       false|
        |  6|{null, f, false}|{null, 3, false}|{false, true}|       false|
        +---+----------------+----------------+-------------+------------+
        <BLANKLINE>
        >>> diff_result.top_per_col_state_df.show()
        +-----------+-------------+----------+-----------+---+---------------+-------+
        |column_name|        state|left_value|right_value| nb|col_state_total|row_num|
        +-----------+-------------+----------+-----------+---+---------------+-------+
        |         c1|    no_change|         b|          b|  3|              4|      1|
        |         c1|    no_change|         a|          a|  1|              4|      2|
        |         c1| only_in_left|         c|       null|  1|              1|      1|
        |         c1|only_in_right|      null|          f|  1|              1|      1|
        |         c2|      changed|         2|          4|  2|              3|      1|
        |         c2|      changed|         2|          3|  1|              3|      2|
        |         c2|    no_change|         1|          1|  1|              1|      1|
        |         c2| only_in_left|         3|       null|  1|              1|      1|
        |         c2|only_in_right|      null|          3|  1|              1|      1|
        |         id|    no_change|         1|          1|  1|              4|      1|
        |         id|    no_change|         2|          2|  1|              4|      2|
        |         id|    no_change|         3|          3|  1|              4|      3|
        |         id|    no_change|         4|          4|  1|              4|      4|
        |         id| only_in_left|         5|       null|  1|              1|      1|
        |         id|only_in_right|      null|          6|  1|              1|      1|
        +-----------+-------------+----------+-----------+---+---------------+-------+
        <BLANKLINE>

        >>> diff_per_col_df = get_diff_per_col_df(diff_result, max_nb_rows_per_col_state=10)
        >>> from spark_frame import nested
        >>> nested.print_schema(diff_per_col_df)
        root
         |-- column_number: integer (nullable = true)
         |-- column_name: string (nullable = true)
         |-- counts.total: integer (nullable = false)
         |-- counts.changed: long (nullable = false)
         |-- counts.no_change: long (nullable = false)
         |-- counts.only_in_left: long (nullable = false)
         |-- counts.only_in_right: long (nullable = false)
         |-- diff.changed!.left_value: string (nullable = true)
         |-- diff.changed!.right_value: string (nullable = true)
         |-- diff.changed!.nb: long (nullable = false)
         |-- diff.no_change!.value: string (nullable = true)
         |-- diff.no_change!.nb: long (nullable = false)
         |-- diff.only_in_left!.value: string (nullable = true)
         |-- diff.only_in_left!.nb: long (nullable = false)
         |-- diff.only_in_right!.value: string (nullable = true)
         |-- diff.only_in_right!.nb: long (nullable = false)
        <BLANKLINE>
        >>> diff_per_col_df.show(truncate=False)  # noqa: E501
        +-------------+-----------+---------------+----------------------------------------------------------+
        |column_number|column_name|counts         |diff                                                      |
        +-------------+-----------+---------------+----------------------------------------------------------+
        |0            |id         |{6, 0, 4, 1, 1}|{[], [{1, 1}, {2, 1}, {3, 1}, {4, 1}], [{5, 1}], [{6, 1}]}|
        |1            |c1         |{6, 0, 4, 1, 1}|{[], [{b, 3}, {a, 1}], [{c, 1}], [{f, 1}]}                |
        |2            |c2         |{6, 3, 1, 1, 1}|{[{2, 4, 2}, {2, 3, 1}], [{1, 1}], [{3, 1}], [{3, 1}]}    |
        +-------------+-----------+---------------+----------------------------------------------------------+
        <BLANKLINE>

        The following test demonstrates that the arrays in `diff`
        are not guaranteed to be sorted by decreasing frequency
        >>> get_diff_per_col_df(diff_result,
        ...     max_nb_rows_per_col_state=10,
        ...     top_per_col_state_df=diff_result.top_per_col_state_df.orderBy("nb")
        ... ).show(truncate=False)  # noqa: E501
        +-------------+-----------+---------------+----------------------------------------------------------+
        |column_number|column_name|counts         |diff                                                      |
        +-------------+-----------+---------------+----------------------------------------------------------+
        |0            |id         |{6, 0, 4, 1, 1}|{[], [{1, 1}, {2, 1}, {3, 1}, {4, 1}], [{5, 1}], [{6, 1}]}|
        |1            |c1         |{6, 0, 4, 1, 1}|{[], [{a, 1}, {b, 3}], [{c, 1}], [{f, 1}]}                |
        |2            |c2         |{6, 3, 1, 1, 1}|{[{2, 3, 1}, {2, 4, 2}], [{1, 1}], [{3, 1}], [{3, 1}]}    |
        +-------------+-----------+---------------+----------------------------------------------------------+
        <BLANKLINE>
    """
    diff_stats = diff_result.diff_stats
    if top_per_col_state_df is None:
        _top_per_col_state_df = diff_result.top_per_col_state_df
    else:
        _top_per_col_state_df = top_per_col_state_df
    spark = _top_per_col_state_df.sparkSession
    columns = diff_result.diff_df.columns[:-2]
    pivoted_df = _get_pivoted_df(_top_per_col_state_df, max_nb_rows_per_col_state)
    col_df = _get_col_df(columns, spark)
    df = _format_diff_per_col_df(pivoted_df, col_df, diff_stats)
    return df
