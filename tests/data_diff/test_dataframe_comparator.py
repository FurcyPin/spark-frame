import warnings
from typing import TYPE_CHECKING

from pyspark.sql import SparkSession

from spark_frame.data_diff.dataframe_comparator import DataframeComparator
from spark_frame.data_diff.diff_stats import DiffStats

if TYPE_CHECKING:
    from spark_frame.data_diff.diff_results import DiffResult


def test_dataframe_comparator_is_deprecated(spark: SparkSession):
    df = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(1 as col1, "a" as col2),
            STRUCT(2 as col1, "b" as col2),
            STRUCT(3 as col1, NULL as col2)
        ))
        """,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df_comparator = DataframeComparator()
        assert "deprecated" in str(w[-1].message)

    # Check that the deprecated method still runs
    diff_result: DiffResult = df_comparator.compare_df(df, df)
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
