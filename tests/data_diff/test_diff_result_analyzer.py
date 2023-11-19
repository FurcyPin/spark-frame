from pyspark.sql import SparkSession

from spark_frame.data_diff import compare_dataframes
from spark_frame.data_diff.diff_format_options import DiffFormatOptions
from spark_frame.data_diff.diff_results import DiffResult
from spark_frame.data_diff.diff_stats import DiffStats
from spark_frame.utils import show_string, strip_margin


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
        """
    )
    df2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a1" as col1),
            STRUCT(2 as id, "b1" as col1),
            STRUCT(3 as id, "c1" as col1)
        ))
        """
    )
    join_cols = ["id"]
    diff_result: DiffResult = compare_dataframes(df1, df2, join_cols)
    expected_diff_stats = DiffStats(
        total=3, no_change=0, changed=3, in_left=3, in_right=3, only_in_left=0, only_in_right=0
    )
    assert diff_result.same_schema is True
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats

    assert show_string(diff_result.top_per_col_state_df, truncate=False) == strip_margin(
        """
        |+-----------+---------+----------+-----------+---+---------------+-------+
        ||column_name|state    |left_value|right_value|nb |col_state_total|row_num|
        |+-----------+---------+----------+-----------+---+---------------+-------+
        ||col1       |changed  |a         |a1         |1  |3              |1      |
        ||col1       |changed  |b         |b1         |1  |3              |2      |
        ||col1       |changed  |c         |c1         |1  |3              |3      |
        ||id         |no_change|1         |1          |1  |3              |1      |
        ||id         |no_change|2         |2          |1  |3              |2      |
        ||id         |no_change|3         |3          |1  |3              |3      |
        |+-----------+---------+----------+-----------+---+---------------+-------+
        |"""
    )
    from spark_frame.data_diff.diff_result_analyzer import DiffResultAnalyzer

    analyzer = DiffResultAnalyzer(DiffFormatOptions(nb_diffed_rows=1))
    diff_per_col_df = analyzer._get_diff_per_col_df(diff_result)
    assert show_string(diff_per_col_df, truncate=False) == strip_margin(
        """
        |+-------------+-----------+---------------+--------------------------+
        ||column_number|column_name|counts         |diff                      |
        |+-------------+-----------+---------------+--------------------------+
        ||0            |id         |{3, 0, 3, 0, 0}|{[], [{1, 1}], [], []}    |
        ||1            |col1       |{3, 3, 0, 0, 0}|{[{a, a1, 1}], [], [], []}|
        |+-------------+-----------+---------------+--------------------------+
        |"""
    )


def test_when_we_have_values_that_are_longer_than_max_string_length(spark: SparkSession):
    """
    GIVEN two DataFrames differing with column values that are longer than max_string_length
          but are identical on the first "max_string_length" characters
    WHEN we compare them
    THEN the stats should be correct
    """
    df1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "aaxxxx" as col1),
            STRUCT(2 as id, "aayyyy" as col1),
            STRUCT(3 as id, "aazzzz" as col1)
        ))
        """
    )
    df2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "AAXXXX" as col1),
            STRUCT(2 as id, "AAYYYY" as col1),
            STRUCT(3 as id, "AAZZZZ" as col1)
        ))
        """
    )
    join_cols = ["id"]
    diff_result: DiffResult = compare_dataframes(df1, df2, join_cols)
    expected_diff_stats = DiffStats(
        total=3, no_change=0, changed=3, in_left=3, in_right=3, only_in_left=0, only_in_right=0
    )
    assert diff_result.same_schema is True
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats

    from spark_frame.data_diff.diff_result_analyzer import DiffResultAnalyzer

    analyzer = DiffResultAnalyzer(DiffFormatOptions(max_string_length=5))
    diff_per_col_df = analyzer._get_diff_per_col_df(diff_result)
    assert show_string(diff_result.top_per_col_state_df, truncate=False) == strip_margin(
        """
        |+-----------+---------+----------+-----------+---+---------------+-------+
        ||column_name|state    |left_value|right_value|nb |col_state_total|row_num|
        |+-----------+---------+----------+-----------+---+---------------+-------+
        ||col1       |changed  |aaxxxx    |AAXXXX     |1  |3              |1      |
        ||col1       |changed  |aayyyy    |AAYYYY     |1  |3              |2      |
        ||col1       |changed  |aazzzz    |AAZZZZ     |1  |3              |3      |
        ||id         |no_change|1         |1          |1  |3              |1      |
        ||id         |no_change|2         |2          |1  |3              |2      |
        ||id         |no_change|3         |3          |1  |3              |3      |
        |+-----------+---------+----------+-----------+---+---------------+-------+
        |"""
    )
    assert show_string(diff_per_col_df.select("column_number", "diff"), truncate=False) == strip_margin(
        """
        |+-------------+-----------------------------------------------------------------------------+
        ||column_number|diff                                                                         |
        |+-------------+-----------------------------------------------------------------------------+
        ||0            |{[], [{1, 1}, {2, 1}, {3, 1}], [], []}                                       |
        ||1            |{[{aaxxxx, AAXXXX, 1}, {aayyyy, AAYYYY, 1}, {aazzzz, AAZZZZ, 1}], [], [], []}|
        |+-------------+-----------------------------------------------------------------------------+
        |"""
    )
