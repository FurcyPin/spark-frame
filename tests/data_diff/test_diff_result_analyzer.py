from typing import TYPE_CHECKING

from pyspark.sql import SparkSession

from spark_frame.data_diff import compare_dataframes
from spark_frame.data_diff.diff_format_options import DiffFormatOptions
from spark_frame.data_diff.diff_stats import DiffStats
from spark_frame.utils import show_string, strip_margin

if TYPE_CHECKING:
    from spark_frame.data_diff.diff_result import DiffResult


def test_when_we_have_more_lines_than_nb_diffed_rows(spark: SparkSession):
    """
    GIVEN two DataFrames differing with more lines than `DiffFormatOptions.nb_diffed_rows`
    WHEN we compare them
    THEN the stats should be correct
    """
    df1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a" as col1),
            STRUCT(2 as id, "b" as col1),
            STRUCT(3 as id, "c" as col1)
        ))
        """,
    )
    df2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a1" as col1),
            STRUCT(2 as id, "b1" as col1),
            STRUCT(3 as id, "c1" as col1)
        ))
        """,
    )
    join_cols = ["id"]
    diff_result: DiffResult = compare_dataframes(df1, df2, join_cols)
    expected_diff_stats = DiffStats(
        total=3,
        no_change=0,
        changed=3,
        in_left=3,
        in_right=3,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is True
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats

    assert show_string(diff_result.top_per_col_state_df, truncate=False) == strip_margin(
        """
        |+-----------+---------+----------+-----------+---+-------+
        ||column_name|state    |left_value|right_value|nb |row_num|
        |+-----------+---------+----------+-----------+---+-------+
        ||col1       |changed  |a         |a1         |1  |1      |
        ||col1       |changed  |b         |b1         |1  |2      |
        ||col1       |changed  |c         |c1         |1  |3      |
        ||id         |no_change|1         |1          |1  |1      |
        ||id         |no_change|2         |2          |1  |2      |
        ||id         |no_change|3         |3          |1  |3      |
        |+-----------+---------+----------+-----------+---+-------+
        |""",
    )
    from spark_frame.data_diff.diff_result_analyzer import DiffResultAnalyzer

    analyzer = DiffResultAnalyzer(DiffFormatOptions(nb_top_values_kept_per_column=1))
    diff_per_col_df = analyzer.get_diff_per_col_df(diff_result)
    assert show_string(diff_per_col_df, truncate=False) == strip_margin(
        """
        |+-------------+-----------+---------------+--------------------------+
        ||column_number|column_name|counts         |diff                      |
        |+-------------+-----------+---------------+--------------------------+
        ||0            |id         |{3, 0, 3, 0, 0}|{[], [{1, 1}], [], []}    |
        ||1            |col1       |{3, 3, 0, 0, 0}|{[{a, a1, 1}], [], [], []}|
        |+-------------+-----------+---------------+--------------------------+
        |""",
    )
