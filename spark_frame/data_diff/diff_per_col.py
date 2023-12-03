from functools import lru_cache
from typing import TYPE_CHECKING, List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from spark_frame import nested
from spark_frame.data_diff.special_characters import (
    _restore_special_characters_from_col,
)

if TYPE_CHECKING:
    from spark_frame.data_diff.diff_results import DiffResult


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
    col_df = spark.createDataFrame(
        list(enumerate(columns)),
        "column_number INT, column_name STRING",
    ).withColumn(
        "column_name",
        _restore_special_characters_from_col(f.col("column_name")),
    )
    return col_df


def _get_pivoted_df(
    top_per_col_state_df: DataFrame,
    max_nb_rows_per_col_state: int,
) -> DataFrame:
    """Pivot the top_per_col_state_df

    Examples:

        >>> from spark_frame.data_diff.diff_results import _get_test_diff_result
        >>> diff_result = _get_test_diff_result()
        >>> diff_result.top_per_col_state_df.show(100)
        +-----------+-------------+----------+-----------+---+---------------+-------+
        |column_name|        state|left_value|right_value| nb|col_state_total|row_num|
        +-----------+-------------+----------+-----------+---+---------------+-------+
        |         c1|    no_change|         b|          b|  3|              4|      1|
        |         c1|    no_change|         a|          a|  1|              4|      2|
        |         c1| only_in_left|         c|       NULL|  1|              1|      1|
        |         c1|only_in_right|      NULL|          f|  1|              1|      1|
        |         c2|      changed|         2|          4|  2|              3|      1|
        |         c2|      changed|         2|          3|  1|              3|      2|
        |         c2|    no_change|         1|          1|  1|              1|      1|
        |         c2| only_in_left|         3|       NULL|  1|              1|      1|
        |         c2|only_in_right|      NULL|          3|  1|              1|      1|
        |         c3| only_in_left|         1|       NULL|  2|              5|      1|
        |         c3| only_in_left|         2|       NULL|  2|              5|      2|
        |         c3| only_in_left|         3|       NULL|  1|              5|      3|
        |         c4|only_in_right|      NULL|          1|  2|              5|      1|
        |         c4|only_in_right|      NULL|          2|  2|              5|      2|
        |         c4|only_in_right|      NULL|          3|  1|              5|      3|
        |         id|    no_change|         1|          1|  1|              4|      1|
        |         id|    no_change|         2|          2|  1|              4|      2|
        |         id|    no_change|         3|          3|  1|              4|      3|
        |         id|    no_change|         4|          4|  1|              4|      4|
        |         id| only_in_left|         5|       NULL|  1|              1|      1|
        |         id|only_in_right|      NULL|          6|  1|              1|      1|
        +-----------+-------------+----------+-----------+---+---------------+-------+
        <BLANKLINE>

        >>> _get_pivoted_df(diff_result.top_per_col_state_df, max_nb_rows_per_col_state=10).show(truncate=False)
        +-----------+----------+----------------------+------------+--------------------------------------------+---------------+------------------------------------------+----------------+------------------------------------------+
        |column_name|changed_nb|changed_diff          |no_change_nb|no_change_diff                              |only_in_left_nb|only_in_left_diff                         |only_in_right_nb|only_in_right_diff                        |
        +-----------+----------+----------------------+------------+--------------------------------------------+---------------+------------------------------------------+----------------+------------------------------------------+
        |c1         |NULL      |[]                    |4           |[{b, b, 3}, {a, a, 1}]                      |1              |[{c, NULL, 1}]                            |1               |[{NULL, f, 1}]                            |
        |c4         |NULL      |[]                    |NULL        |[]                                          |NULL           |[]                                        |5               |[{NULL, 1, 2}, {NULL, 2, 2}, {NULL, 3, 1}]|
        |c3         |NULL      |[]                    |NULL        |[]                                          |5              |[{1, NULL, 2}, {2, NULL, 2}, {3, NULL, 1}]|NULL            |[]                                        |
        |c2         |3         |[{2, 4, 2}, {2, 3, 1}]|1           |[{1, 1, 1}]                                 |1              |[{3, NULL, 1}]                            |1               |[{NULL, 3, 1}]                            |
        |id         |NULL      |[]                    |4           |[{1, 1, 1}, {2, 2, 1}, {3, 3, 1}, {4, 4, 1}]|1              |[{5, NULL, 1}]                            |1               |[{NULL, 6, 1}]                            |
        +-----------+----------+----------------------+------------+--------------------------------------------+---------------+------------------------------------------+----------------+------------------------------------------+
        <BLANKLINE>
    """  # noqa: E501
    pivoted_df = (
        top_per_col_state_df.groupBy("column_name")
        .pivot(
            "state",
            values=["changed", "no_change", "only_in_left", "only_in_right"],
        )
        .agg(
            f.sum("nb").alias("nb"),
            f.expr(
                f"""
            ARRAY_AGG(
                CASE WHEN row_num <= {max_nb_rows_per_col_state} THEN STRUCT(left_value, right_value, nb) END
            )""",
            ).alias("diff"),
        )
    )
    return pivoted_df


def _format_diff_per_col_df(pivoted_df: DataFrame, col_df: DataFrame) -> DataFrame:
    """

    Examples

        >>> from spark_frame.data_diff.diff_results import _get_test_diff_result
        >>> diff_result = _get_test_diff_result()
        >>> spark = diff_result.top_per_col_state_df.sparkSession
        >>> columns = diff_result.schema_diff_result.column_names
        >>> col_df = _get_col_df(columns, spark)
        >>> pivoted_df = _get_pivoted_df(diff_result.top_per_col_state_df, max_nb_rows_per_col_state=10)
        >>> col_df.show()
        +-------------+-----------+
        |column_number|column_name|
        +-------------+-----------+
        |            0|         id|
        |            1|         c1|
        |            2|         c2|
        |            3|         c3|
        |            4|         c4|
        +-------------+-----------+
        <BLANKLINE>
        >>> pivoted_df.show(truncate=False)
        +-----------+----------+----------------------+------------+--------------------------------------------+---------------+------------------------------------------+----------------+------------------------------------------+
        |column_name|changed_nb|changed_diff          |no_change_nb|no_change_diff                              |only_in_left_nb|only_in_left_diff                         |only_in_right_nb|only_in_right_diff                        |
        +-----------+----------+----------------------+------------+--------------------------------------------+---------------+------------------------------------------+----------------+------------------------------------------+
        |c1         |NULL      |[]                    |4           |[{b, b, 3}, {a, a, 1}]                      |1              |[{c, NULL, 1}]                            |1               |[{NULL, f, 1}]                            |
        |c4         |NULL      |[]                    |NULL        |[]                                          |NULL           |[]                                        |5               |[{NULL, 1, 2}, {NULL, 2, 2}, {NULL, 3, 1}]|
        |c3         |NULL      |[]                    |NULL        |[]                                          |5              |[{1, NULL, 2}, {2, NULL, 2}, {3, NULL, 1}]|NULL            |[]                                        |
        |c2         |3         |[{2, 4, 2}, {2, 3, 1}]|1           |[{1, 1, 1}]                                 |1              |[{3, NULL, 1}]                            |1               |[{NULL, 3, 1}]                            |
        |id         |NULL      |[]                    |4           |[{1, 1, 1}, {2, 2, 1}, {3, 3, 1}, {4, 4, 1}]|1              |[{5, NULL, 1}]                            |1               |[{NULL, 6, 1}]                            |
        +-----------+----------+----------------------+------------+--------------------------------------------+---------------+------------------------------------------+----------------+------------------------------------------+
        <BLANKLINE>
        >>> diff_per_col_df = _format_diff_per_col_df(pivoted_df, col_df)
        >>> nested.print_schema(diff_per_col_df)
        root
         |-- column_number: integer (nullable = true)
         |-- column_name: string (nullable = true)
         |-- counts.total: long (nullable = false)
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
        |3            |c3         |{5, 0, 0, 5, 0}|{[], [], [{1, 2}, {2, 2}, {3, 1}], []}                    |
        |4            |c4         |{5, 0, 0, 0, 5}|{[], [], [], [{1, 2}, {2, 2}, {3, 1}]}                    |
        +-------------+-----------+---------------+----------------------------------------------------------+
        <BLANKLINE>
    """  # noqa: E501
    coalesced_df = col_df.join(pivoted_df, "column_name", "left").withColumns(
        {
            "changed_nb": f.coalesce(f.col("changed_nb"), f.lit(0)),
            "no_change_nb": f.coalesce(f.col("no_change_nb"), f.lit(0)),
            "only_in_left_nb": f.coalesce(f.col("only_in_left_nb"), f.lit(0)),
            "only_in_right_nb": f.coalesce(f.col("only_in_right_nb"), f.lit(0)),
            "changed_diff": f.coalesce(f.col("changed_diff"), f.array()),
            "no_change_diff": f.coalesce(f.col("no_change_diff"), f.array()),
            "only_in_left_diff": f.coalesce(f.col("only_in_left_diff"), f.array()),
            "only_in_right_diff": f.coalesce(f.col("only_in_right_diff"), f.array()),
        },
    )
    total_col = f.col("changed_nb") + f.col("no_change_nb") + f.col("only_in_left_nb") + f.col("only_in_right_nb")
    renamed_df = coalesced_df.select(
        f.col("column_number"),
        f.col("column_name"),
        f.struct(
            total_col.alias("total"),
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


@lru_cache()
def _get_diff_per_col_df_with_cache(diff_result: "DiffResult", max_nb_rows_per_col_state: int) -> DataFrame:
    from spark_frame.data_diff.diff_per_col import _get_diff_per_col_df

    return _get_diff_per_col_df(
        top_per_col_state_df=diff_result.top_per_col_state_df,
        columns=diff_result.schema_diff_result.column_names,
        max_nb_rows_per_col_state=max_nb_rows_per_col_state,
    ).localCheckpoint()


def _get_diff_per_col_df(
    top_per_col_state_df: DataFrame,
    columns: List[str],
    max_nb_rows_per_col_state: int,
) -> DataFrame:
    """Given a top_per_col_state_df, return a DataFrame that gives for each column and each
    column state (changed, no_change, only_in_left, only_in_right) the total number of occurences
    and the most frequent occurrences.

    !!! warning
        The arrays contained in the field `diff` are NOT guaranteed to be sorted,
        and Spark currently does not provide any way to perform a sort_by on an ARRAY<STRUCT>.

    Args:
        top_per_col_state_df: A DataFrame with the following columns
            - column_name
            - state
            - left_value
            - right_value
            - nb
            - col_state_total
            - row_num
        columns: The list of column names to use. The column ordering given by this list is preserved.
        max_nb_rows_per_col_state: The maximal size of the arrays in `diff`

    Returns:
        A DataFrame with the following schema:

            root
             |-- column_number: integer (nullable = true)
             |-- column_name: string (nullable = true)
             |-- counts.total: long (nullable = false)
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
        >>> diff_df = diff_result.diff_df_shards[""]
        >>> diff_df.show(truncate=False)
        +-----------------------------+-----------------------------+-----------------------------+---------------------------------+---------------------------------+-------------+------------+
        |id                           |c1                           |c2                           |c3                               |c4                               |__EXISTS__   |__IS_EQUAL__|
        +-----------------------------+-----------------------------+-----------------------------+---------------------------------+---------------------------------+-------------+------------+
        |{1, 1, true, true, true}     |{a, a, true, true, true}     |{1, 1, true, true, true}     |{1, NULL, false, true, false}    |{NULL, 1, false, false, true}    |{true, true} |true        |
        |{2, 2, true, true, true}     |{b, b, true, true, true}     |{2, 3, false, true, true}    |{1, NULL, false, true, false}    |{NULL, 1, false, false, true}    |{true, true} |false       |
        |{3, 3, true, true, true}     |{b, b, true, true, true}     |{2, 4, false, true, true}    |{2, NULL, false, true, false}    |{NULL, 2, false, false, true}    |{true, true} |false       |
        |{4, 4, true, true, true}     |{b, b, true, true, true}     |{2, 4, false, true, true}    |{2, NULL, false, true, false}    |{NULL, 2, false, false, true}    |{true, true} |false       |
        |{5, NULL, false, true, false}|{c, NULL, false, true, false}|{3, NULL, false, true, false}|{3, NULL, false, true, false}    |{NULL, NULL, false, false, false}|{true, false}|false       |
        |{NULL, 6, false, false, true}|{NULL, f, false, false, true}|{NULL, 3, false, false, true}|{NULL, NULL, false, false, false}|{NULL, 3, false, false, true}    |{false, true}|false       |
        +-----------------------------+-----------------------------+-----------------------------+---------------------------------+---------------------------------+-------------+------------+
        <BLANKLINE>
        >>> diff_result.top_per_col_state_df.show(100)
        +-----------+-------------+----------+-----------+---+---------------+-------+
        |column_name|        state|left_value|right_value| nb|col_state_total|row_num|
        +-----------+-------------+----------+-----------+---+---------------+-------+
        |         c1|    no_change|         b|          b|  3|              4|      1|
        |         c1|    no_change|         a|          a|  1|              4|      2|
        |         c1| only_in_left|         c|       NULL|  1|              1|      1|
        |         c1|only_in_right|      NULL|          f|  1|              1|      1|
        |         c2|      changed|         2|          4|  2|              3|      1|
        |         c2|      changed|         2|          3|  1|              3|      2|
        |         c2|    no_change|         1|          1|  1|              1|      1|
        |         c2| only_in_left|         3|       NULL|  1|              1|      1|
        |         c2|only_in_right|      NULL|          3|  1|              1|      1|
        |         c3| only_in_left|         1|       NULL|  2|              5|      1|
        |         c3| only_in_left|         2|       NULL|  2|              5|      2|
        |         c3| only_in_left|         3|       NULL|  1|              5|      3|
        |         c4|only_in_right|      NULL|          1|  2|              5|      1|
        |         c4|only_in_right|      NULL|          2|  2|              5|      2|
        |         c4|only_in_right|      NULL|          3|  1|              5|      3|
        |         id|    no_change|         1|          1|  1|              4|      1|
        |         id|    no_change|         2|          2|  1|              4|      2|
        |         id|    no_change|         3|          3|  1|              4|      3|
        |         id|    no_change|         4|          4|  1|              4|      4|
        |         id| only_in_left|         5|       NULL|  1|              1|      1|
        |         id|only_in_right|      NULL|          6|  1|              1|      1|
        +-----------+-------------+----------+-----------+---+---------------+-------+
        <BLANKLINE>

        >>> diff_per_col_df = _get_diff_per_col_df(
        ...     diff_result.top_per_col_state_df,
        ...     diff_result.schema_diff_result.column_names,
        ...     max_nb_rows_per_col_state=10)
        >>> from spark_frame import nested
        >>> nested.print_schema(diff_per_col_df)
        root
         |-- column_number: integer (nullable = true)
         |-- column_name: string (nullable = true)
         |-- counts.total: long (nullable = false)
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
        |3            |c3         |{5, 0, 0, 5, 0}|{[], [], [{1, 2}, {2, 2}, {3, 1}], []}                    |
        |4            |c4         |{5, 0, 0, 0, 5}|{[], [], [], [{1, 2}, {2, 2}, {3, 1}]}                    |
        +-------------+-----------+---------------+----------------------------------------------------------+
        <BLANKLINE>

        The following test demonstrates that the arrays in `diff`
        are not guaranteed to be sorted by decreasing frequency
        >>> _get_diff_per_col_df(
        ...     diff_result.top_per_col_state_df.orderBy("nb"),
        ...     diff_result.schema_diff_result.column_names,
        ...     max_nb_rows_per_col_state=10
        ... ).show(truncate=False)
        +-------------+-----------+---------------+----------------------------------------------------------+
        |column_number|column_name|counts         |diff                                                      |
        +-------------+-----------+---------------+----------------------------------------------------------+
        |0            |id         |{6, 0, 4, 1, 1}|{[], [{1, 1}, {2, 1}, {3, 1}, {4, 1}], [{5, 1}], [{6, 1}]}|
        |1            |c1         |{6, 0, 4, 1, 1}|{[], [{a, 1}, {b, 3}], [{c, 1}], [{f, 1}]}                |
        |2            |c2         |{6, 3, 1, 1, 1}|{[{2, 3, 1}, {2, 4, 2}], [{1, 1}], [{3, 1}], [{3, 1}]}    |
        |3            |c3         |{5, 0, 0, 5, 0}|{[], [], [{3, 1}, {1, 2}, {2, 2}], []}                    |
        |4            |c4         |{5, 0, 0, 0, 5}|{[], [], [], [{3, 1}, {1, 2}, {2, 2}]}                    |
        +-------------+-----------+---------------+----------------------------------------------------------+
        <BLANKLINE>
    """  # noqa: E501
    spark = top_per_col_state_df.sparkSession
    pivoted_df = _get_pivoted_df(top_per_col_state_df, max_nb_rows_per_col_state)
    col_df = _get_col_df(columns, spark)
    df = _format_diff_per_col_df(pivoted_df, col_df)
    return df
