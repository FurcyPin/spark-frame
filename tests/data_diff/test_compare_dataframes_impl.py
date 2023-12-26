import inspect

from pyspark.sql import SparkSession

from spark_frame.data_diff.compare_dataframes_impl import _automatically_infer_join_col, compare_dataframes
from spark_frame.data_diff.diff_result_analyzer import DiffResultAnalyzer
from spark_frame.data_diff.diff_results import DiffResult
from spark_frame.data_diff.diff_stats import DiffStats
from spark_frame.utils import show_string, strip_margin


def export_diff_result_to_html(diff_result: DiffResult):
    test_method_name = inspect.stack()[1][3]
    export_path = "test_working_dir/" + test_method_name + ".html"
    diff_result.export_to_html(title=test_method_name, output_file_path=export_path)


def test_compare_df_with_simplest(spark: SparkSession):
    df = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as col1, "a" as col2),
            STRUCT(2 as col1, "b" as col2),
            STRUCT(3 as col1, NULL as col2)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df, df)
    expected_diff_stats = DiffStats(
        total=3,
        no_change=3,
        changed=0,
        in_left=3,
        in_right=3,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is True
    assert diff_result.is_ok is True
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_ordering(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a" as name),
            STRUCT(2 as id, "b" as name),
            STRUCT(3 as id, "c" as name)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(2 as id, "b" as name),
            STRUCT(3 as id, "c" as name),
            STRUCT(1 as id, "a" as name)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2)
    expected_diff_stats = DiffStats(
        total=3,
        no_change=3,
        changed=0,
        in_left=3,
        in_right=3,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is True
    assert diff_result.is_ok is True
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_empty_dataframes(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a" as name),
            STRUCT(2 as id, "b" as name),
            STRUCT(3 as id, "c" as name)
        )) LIMIT 0
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a" as name),
            STRUCT(2 as id, "b" as name),
            STRUCT(3 as id, "d" as name)
        )) LIMIT 0
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2, join_cols=["id"])
    expected_diff_stats = DiffStats(
        total=0,
        no_change=0,
        changed=0,
        in_left=0,
        in_right=0,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is True
    assert diff_result.is_ok is True
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_different_keys(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a" as name),
            STRUCT(2 as id, "b" as name),
            STRUCT(3 as id, "c" as name)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a" as name),
            STRUCT(2 as id, "b" as name),
            STRUCT(4 as id, "c" as name)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2, join_cols=["id"])
    expected_diff_stats = DiffStats(
        total=4,
        no_change=2,
        changed=0,
        in_left=3,
        in_right=3,
        only_in_left=1,
        only_in_right=1,
    )
    assert diff_result.same_schema is True
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_structs(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, STRUCT(1 as a, 2 as b, 3 as c) as a),
            STRUCT(2 as id, STRUCT(1 as a, 2 as b, 3 as c) as a),
            STRUCT(3 as id, STRUCT(1 as a, 2 as b, 3 as c) as a)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, STRUCT(1 as a, 2 as b, 3 as c) as a),
            STRUCT(2 as id, STRUCT(1 as a, 2 as b, 3 as c) as a),
            STRUCT(3 as id, STRUCT(1 as a, 2 as b, 4 as c) as a)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2, join_cols=["id"])
    expected_diff_stats = DiffStats(
        total=3,
        no_change=2,
        changed=1,
        in_left=3,
        in_right=3,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is True
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)
    analyzer = DiffResultAnalyzer()
    diff_per_col_df = analyzer.get_diff_per_col_df(diff_result)
    # We make sure that the displayed column name is 'a.c' and not 'a__STRUCT__c'
    assert diff_per_col_df.collect()[3].asDict().get("column_name") == "a.c"


def test_compare_df_with_struct_and_different_schemas(spark: SparkSession):
    """A bug was happening when dataframes had different schemas"""
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, STRUCT(1 as a, 1 as b) as s1),
            STRUCT(2 as id, STRUCT(1 as a, 1 as b) as s1),
            STRUCT(3 as id, STRUCT(1 as a, 1 as b) as s1)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, STRUCT(1 as a, 1 as c) as s1),
            STRUCT(2 as id, STRUCT(2 as a, 1 as c) as s1),
            STRUCT(3 as id, STRUCT(1 as a, 1 as c) as s1)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2, join_cols=["id"])
    expected_diff_stats = DiffStats(
        total=3,
        no_change=2,
        changed=1,
        in_left=3,
        in_right=3,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is False
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_arrays(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, ARRAY(1, 2, 3) as a),
            STRUCT(2 as id, ARRAY(1, 2, 3) as a),
            STRUCT(3 as id, ARRAY(1, 2, 3) as a)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, ARRAY(1, 2, 3) as a),
            STRUCT(2 as id, ARRAY(3, 2, 1) as a),
            STRUCT(3 as id, ARRAY(3, 1, 2) as a)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2)
    expected_diff_stats = DiffStats(
        total=3,
        no_change=3,
        changed=0,
        in_left=3,
        in_right=3,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is True
    assert diff_result.is_ok is True
    assert diff_result.diff_stats_shards[""] == expected_diff_stats

    diff_per_col_df = diff_result.get_diff_per_col_df(100)
    columns = diff_per_col_df.select("column_number", "column_name").sort("column_number")
    # The columns should be displayed in the same order as in the original DataFrames, join_cols should not come first.
    assert show_string(columns) == strip_margin(
        """
        |+-------------+-----------+
        ||column_number|column_name|
        |+-------------+-----------+
        ||            0|         id|
        ||            1|          a|
        |+-------------+-----------+
        |""",
    )
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_empty_and_null_arrays(spark: SparkSession):
    """
    GIVEN two DataFrames, one with an empty array, the other with a null array
    WHEN we compare them
    THEN a difference should be found

    Explanation: This test differs from bigquery-frame. In BigQuery, empty arrays and null arrays become equal
                 when persisted
    """
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, CAST(ARRAY() AS ARRAY<INT>) as a)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, CAST(NULL AS ARRAY<INT>) as a)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2)
    expected_diff_stats = DiffStats(
        total=1,
        no_change=0,
        changed=1,
        in_left=1,
        in_right=1,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is True
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_missing_empty_and_null_arrays(spark: SparkSession):
    """
    GIVEN two DataFrames, one with a null and an empty array, the other without those rows
    WHEN we compare them
    THEN differences should be found

    Explanation: This test differs from bigquery-frame, where empty arrays and null arrays become equal after they
                 are being persisted
    """
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, ARRAY(1) as a),
            STRUCT(2 as id, CAST(ARRAY() AS ARRAY<INT>) as a),
            STRUCT(3 as id, CAST(NULL AS ARRAY<INT>) as a)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, ARRAY(2) as a)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2, join_cols=["id"])
    expected_diff_stats = DiffStats(
        total=3,
        no_change=0,
        changed=1,
        in_left=3,
        in_right=1,
        only_in_left=2,
        only_in_right=0,
    )
    assert diff_result.same_schema is True
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)
    analyzer = DiffResultAnalyzer()
    diff_per_col_df = analyzer.get_diff_per_col_df(diff_result)
    assert diff_per_col_df.count() == 2


def test_compare_df_with_arrays_of_structs_ok(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, ARRAY(STRUCT(1 as a, "a" as b), STRUCT(2 as a, "a" as b), STRUCT(3 as a, "a" as b)) as a),
            STRUCT(2 as id, ARRAY(STRUCT(1 as a, "b" as b), STRUCT(2 as a, "b" as b), STRUCT(3 as a, "b" as b)) as a),
            STRUCT(3 as id, ARRAY(STRUCT(1 as a, "c" as b), STRUCT(2 as a, "c" as b), STRUCT(3 as a, "c" as b)) as a)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, ARRAY(STRUCT("a" as b, 3 as a), STRUCT("a" as b, 1 as a), STRUCT("a" as b, 2 as a)) as a),
            STRUCT(2 as id, ARRAY(STRUCT("b" as b, 1 as a), STRUCT("b" as b, 3 as a), STRUCT("b" as b, 2 as a)) as a),
            STRUCT(3 as id, ARRAY(STRUCT("c" as b, 3 as a), STRUCT("c" as b, 2 as a), STRUCT("c" as b, 1 as a)) as a)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2)
    expected_diff_stats = DiffStats(
        total=3,
        no_change=3,
        changed=0,
        in_left=3,
        in_right=3,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is False
    assert diff_result.same_data is True
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_arrays_of_structs_not_ok(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, ARRAY(STRUCT(1 as a, "a" as b), STRUCT(2 as a, "a" as b), STRUCT(3 as a, "a" as b)) as a),
            STRUCT(2 as id, ARRAY(STRUCT(1 as a, "b" as b), STRUCT(2 as a, "b" as b), STRUCT(3 as a, "b" as b)) as a),
            STRUCT(3 as id, ARRAY(STRUCT(1 as a, "c" as b), STRUCT(2 as a, "c" as b), STRUCT(3 as a, "c" as b)) as a)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, ARRAY(STRUCT("a" as b, 3 as a), STRUCT("a" as b, 1 as a), STRUCT("a" as b, 2 as a)) as a),
            STRUCT(2 as id, ARRAY(STRUCT("b" as b, 1 as a), STRUCT("b" as b, 3 as a), STRUCT("b" as b, 2 as a)) as a),
            STRUCT(3 as id, ARRAY(STRUCT("c" as b, 3 as a), STRUCT("c" as b, 2 as a), STRUCT("d" as b, 1 as a)) as a)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2)
    expected_diff_stats = DiffStats(
        total=3,
        no_change=2,
        changed=1,
        in_left=3,
        in_right=3,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is False
    assert diff_result.same_data is False
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_multiple_arrays_of_structs_not_ok(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(
                1 as id1,
                1 as id2,
                ARRAY(
                    STRUCT(
                        1 as id,
                        "a" as a,
                        ARRAY(STRUCT(1 as id, "a" as a), STRUCT(2 as id, "b" as a)) as ss
                    ),
                    STRUCT(
                        2 as id,
                        "a" as a,
                        ARRAY(STRUCT(1 as id, "a" as a), STRUCT(2 as id, "b" as a)) as ss
                    ),
                    STRUCT(
                        3 as id,
                        "a" as a,
                        ARRAY(STRUCT(1 as id, "a" as a), STRUCT(2 as id, "b" as a)) as ss
                    )
                ) as s1,
                ARRAY(
                    STRUCT(1 as id, "a" as a, ARRAY(STRUCT(1 as id, "b" as a)) as ss),
                    STRUCT(2 as id, "a" as a, ARRAY(STRUCT(1 as id, "b" as a)) as ss),
                    STRUCT(3 as id, "a" as a, ARRAY(STRUCT(1 as id, "b" as a)) as ss)
                ) as s2
            ),
            STRUCT(
                2 as id1,
                2 as id2,
                ARRAY(
                    STRUCT(
                        1 as id,
                        "a" as a,
                        ARRAY(STRUCT(1 as id, "a" as a), STRUCT(2 as id, "b" as a)) as ss
                    ),
                    STRUCT(
                        2 as id,
                        "a" as a,
                        ARRAY(STRUCT(1 as id, "a" as a), STRUCT(2 as id, "b" as a)) as ss
                    ),
                    STRUCT(
                        3 as id,
                        "a" as a,
                        ARRAY(STRUCT(1 as id, "a" as a), STRUCT(2 as id, "b" as a)) as ss
                    )
                ) as s1,
                ARRAY(
                    STRUCT(1 as id, "b" as a, ARRAY(STRUCT(1 as id, "b" as a)) as ss),
                    STRUCT(2 as id, "b" as a, ARRAY(STRUCT(1 as id, "b" as a)) as ss),
                    STRUCT(3 as id, "b" as a, ARRAY(STRUCT(1 as id, "b" as a)) as ss)
                ) as s2
            )
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(
                1 as id1,
                1 as id2,
                ARRAY(
                    STRUCT(
                        1 as id,
                        "a" as a,
                        ARRAY(STRUCT(1 as id, "a" as a), STRUCT(2 as id, "b" as a)) as ss
                    ),
                    STRUCT(
                        2 as id,
                        "a" as a,
                        ARRAY(STRUCT(1 as id, "a" as a), STRUCT(2 as id, "b" as a)) as ss
                    ),
                    STRUCT(
                        3 as id,
                        "a" as a,
                        ARRAY(STRUCT(1 as id, "a" as a), STRUCT(2 as id, "c" as a)) as ss
                    )
                ) as s1,
                ARRAY(
                    STRUCT(1 as id, "a" as a, ARRAY(STRUCT(1 as id, "b" as a)) as ss),
                    STRUCT(2 as id, "a" as a, ARRAY(STRUCT(1 as id, "b" as a)) as ss),
                    STRUCT(3 as id, "a" as a, ARRAY(STRUCT(1 as id, "b" as a)) as ss)
                ) as s2
            ),
            STRUCT(
                2 as id1,
                2 as id2,
                ARRAY(
                    STRUCT(
                        1 as id,
                        "a" as a,
                        ARRAY(STRUCT(1 as id, "a" as a), STRUCT(2 as id, "b" as a)) as ss
                    ),
                    STRUCT(
                        2 as id,
                        "a" as a,
                        ARRAY(STRUCT(1 as id, "a" as a), STRUCT(2 as id, "b" as a)) as ss
                    ),
                    STRUCT(
                        3 as id,
                        "a" as a,
                        ARRAY(STRUCT(1 as id, "a" as a), STRUCT(3 as id, "b" as a)) as ss
                    )
                ) as s1,
                ARRAY(
                    STRUCT(1 as id, "b" as a, ARRAY(STRUCT(1 as id, "b" as a)) as ss),
                    STRUCT(2 as id, "b" as a, ARRAY(STRUCT(1 as id, "b" as a)) as ss),
                    STRUCT(3 as id, "b" as a, ARRAY(STRUCT(1 as id, "b" as a)) as ss)
                ) as s2
            )
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2, join_cols=["id1", "id2", "s1!.id", "s1!.ss!.id"])
    expected_diff_stats_shards = {
        "": DiffStats(total=2, no_change=2, changed=0, in_left=2, in_right=2, only_in_left=0, only_in_right=0),
        "s1": DiffStats(total=6, no_change=6, changed=0, in_left=6, in_right=6, only_in_left=0, only_in_right=0),
        "s1!.ss": DiffStats(
            total=13,
            no_change=10,
            changed=1,
            in_left=12,
            in_right=12,
            only_in_left=1,
            only_in_right=1,
        ),
    }
    assert diff_result.same_schema is True
    assert diff_result.same_data is False
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards == expected_diff_stats_shards
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_differing_types(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a" as name),
            STRUCT(2 as id, "b" as name),
            STRUCT(3 as id, "c" as name)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1.0 as id, "a" as name),
            STRUCT(2.0 as id, "b" as name),
            STRUCT(3.0 as id, "d" as name)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2)
    expected_diff_stats = DiffStats(
        total=3,
        no_change=2,
        changed=1,
        in_left=3,
        in_right=3,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is False
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_when_flattened_column_name_collision(spark: SparkSession):
    """
    GIVEN a DataFrame with a nested column `s`.`a` and a column `s_a`
    WHEN we run a diff on it
    THEN it should not crash
    """
    df = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, STRUCT(1 as a) as s, 2 as s_a),
            STRUCT(2 as id, STRUCT(2 as a) as s, 3 as s_a),
            STRUCT(3 as id, STRUCT(3 as a) as s, 4 as s_a)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df, df, join_cols=["id"])
    expected_diff_stats = DiffStats(
        total=3,
        no_change=3,
        changed=0,
        in_left=3,
        in_right=3,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is True
    assert diff_result.same_data is True
    assert diff_result.is_ok is True
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_null_join_cols(spark: SparkSession):
    """
    GIVEN two DataFrames
    WHEN we diff them using join_cols that are sometimes null
    THEN the null values should correctly be matched together
    """
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id1, 1 as id2, "a" as name),
            STRUCT(2 as id1, 2 as id2, "b" as name),
            STRUCT(NULL as id1, 3 as id2, "c" as name),
            STRUCT(4 as id1, NULL as id2, "d" as name),
            STRUCT(NULL as id1, NULL as id2, "e1" as name)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id1, 1 as id2, "a" as name),
            STRUCT(2 as id1, 2 as id2, "b" as name),
            STRUCT(NULL as id1, 3 as id2, "c" as name),
            STRUCT(4 as id1, NULL as id2, "d" as name),
            STRUCT(NULL as id1, NULL as id2, "e2" as name)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2, join_cols=["id1", "id2"])
    expected_diff_stats = DiffStats(
        total=5,
        no_change=4,
        changed=1,
        in_left=5,
        in_right=5,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is True
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_disappearing_columns(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a" as col1, "a" as disappearing_col),
            STRUCT(2 as id, "b" as col1, "b" as disappearing_col),
            STRUCT(3 as id, "c" as col1, "c" as disappearing_col)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a" as col1),
            STRUCT(2 as id, "b" as col1),
            STRUCT(3 as id, "c" as col1)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2)
    expected_diff_stats = DiffStats(
        total=3,
        no_change=3,
        changed=0,
        in_left=3,
        in_right=3,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is False
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_appearing_columns(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a" as col1),
            STRUCT(2 as id, "b" as col1),
            STRUCT(3 as id, "c" as col1)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a" as col1, "a" as appearing_col),
            STRUCT(2 as id, "b" as col1, "b" as appearing_col),
            STRUCT(3 as id, "c" as col1, "c" as appearing_col)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2)
    expected_diff_stats = DiffStats(
        total=3,
        no_change=3,
        changed=0,
        in_left=3,
        in_right=3,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is False
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_renamed_columns(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a" as col1, "a" as renamed_col_1),
            STRUCT(2 as id, "b" as col1, "b" as renamed_col_1),
            STRUCT(3 as id, "b" as col1, "b" as renamed_col_1),
            STRUCT(4 as id, "c" as col1, "c" as renamed_col_1)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, "a" as col1, "a" as renamed_col_2),
            STRUCT(2 as id, "b" as col1, "b" as renamed_col_2),
            STRUCT(3 as id, "b" as col1, "b" as renamed_col_2),
            STRUCT(5 as id, "c" as col1, "c" as renamed_col_2)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2)
    expected_diff_stats = DiffStats(
        total=5,
        no_change=3,
        changed=0,
        in_left=4,
        in_right=4,
        only_in_left=1,
        only_in_right=1,
    )
    assert diff_result.same_schema is False
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_compare_df_with_renamed_columns_inside_structs(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, STRUCT("a" as col1, "a" as renamed_col_1) as s),
            STRUCT(2 as id, STRUCT("b" as col1, "b" as renamed_col_1) as s),
            STRUCT(3 as id, STRUCT("b" as col1, "b" as renamed_col_1) as s),
            STRUCT(4 as id, STRUCT("c" as col1, "c" as renamed_col_1) as s)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as id, STRUCT("a" as col1, "a" as renamed_col_2) as s),
            STRUCT(2 as id, STRUCT("b" as col1, "b" as renamed_col_2) as s),
            STRUCT(3 as id, STRUCT("b" as col1, "b" as renamed_col_2) as s),
            STRUCT(5 as id, STRUCT("c" as col1, "c" as renamed_col_2) as s)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2)
    expected_diff_stats = DiffStats(
        total=5,
        no_change=3,
        changed=0,
        in_left=4,
        in_right=4,
        only_in_left=1,
        only_in_right=1,
    )
    assert diff_result.same_schema is False
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats
    diff_result.display()
    export_diff_result_to_html(diff_result)


def test_automatically_infer_join_col(spark: SparkSession):
    """
    - GIVEN two DataFrames with two columns each
    - WHEN one column is unique in both DataFrames and the other is almost unique
    - THEN the unique column should be selected
    """
    left_df = spark.sql(
        """
        WITH T AS (SELECT EXPLODE(SEQUENCE(1, 20)) as id)
        SELECT id as unique_id, id as non_unique_col FROM T
        UNION ALL
        SELECT "101" as unique_id, "101" as non_unique_col
    """,
    )
    right_df = spark.sql(
        """
        WITH T AS (SELECT EXPLODE(SEQUENCE(1, 20)) as id)
        SELECT id as unique_id, id as non_unique_col FROM T
        UNION ALL
        SELECT "101" as unique_id, "1" as non_unique_col
    """,
    )
    join_cols, self_join_growth_estimate = _automatically_infer_join_col(left_df, right_df)
    assert join_cols == "unique_id"
    assert self_join_growth_estimate == 1.0

    # The result should be the same if we exchange left and right
    join_cols, self_join_growth_estimate = _automatically_infer_join_col(right_df, left_df)
    assert join_cols == "unique_id"
    assert self_join_growth_estimate == 1.0


def test_join_cols_should_not_be_displayed_first(spark: SparkSession):
    df_1 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT("a" as name, 1 as id),
            STRUCT("b" as name, 2 as id),
            STRUCT("c" as name, 3 as id)
        ))
        """,
    )
    df_2 = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT("a" as name, 1 as id),
            STRUCT("b" as name, 2 as id),
            STRUCT("d" as name, 3 as id)
        ))
        """,
    )
    diff_result: DiffResult = compare_dataframes(df_1, df_2, join_cols=["id"])
    expected_diff_stats = DiffStats(
        total=3,
        no_change=2,
        changed=1,
        in_left=3,
        in_right=3,
        only_in_left=0,
        only_in_right=0,
    )
    assert diff_result.same_schema is True
    assert diff_result.is_ok is False
    assert diff_result.diff_stats_shards[""] == expected_diff_stats

    diff_per_col_df = diff_result.get_diff_per_col_df(100)
    columns = diff_per_col_df.select("column_number", "column_name").sort("column_number")
    # The columns should be displayed in the same order as in the original DataFrames, join_cols should not come first.
    assert show_string(columns) == strip_margin(
        """
        |+-------------+-----------+
        ||column_number|column_name|
        |+-------------+-----------+
        ||            0|       name|
        ||            1|         id|
        |+-------------+-----------+
        |""",
    )
    diff_result.display()
    export_diff_result_to_html(diff_result)
