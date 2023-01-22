from functools import cached_property
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from spark_frame.data_diff.diff_format_options import DiffFormatOptions
from spark_frame.data_diff.diff_stats import DiffStats
from spark_frame.data_diff.export import export_html_diff_report
from spark_frame.data_diff.package import EXISTS_COL_NAME, IS_EQUAL_COL_NAME, PREDICATES
from spark_frame.data_diff.schema_diff import SchemaDiffResult


class DiffResult:
    def __init__(
        self,
        schema_diff_result: SchemaDiffResult,
        diff_df: DataFrame,
        join_cols: List[str],
        diff_format_options: DiffFormatOptions = DiffFormatOptions(),
    ):
        """Class containing the results of a diff between two DataFrames"""
        self.schema_diff_result = schema_diff_result
        self.diff_df = diff_df
        """A Spark DataFrame with the following schema:

        - All columns from join_cols
        - For all other common columns in the diffed Dataframe:
          a Column `col_name: STRUCT<left_value, right_value, is_equal>`
        - A Column `__EXISTS__: STRUCT<left_value, right_value>`
        - A Column `__IS_EQUAL__: BOOLEAN`
        """
        self.join_cols = join_cols
        """The list of column names to join"""
        self.diff_format_options = diff_format_options
        self._changed_df_shards: Optional[List[DataFrame]] = None

    @property
    def same_schema(self):
        return self.schema_diff_result.same_schema

    @cached_property
    def diff_stats(self):
        return self._compute_diff_stats()

    @property
    def same_data(self):
        return self.diff_stats.same_data

    @property
    def is_ok(self):
        return self.same_schema and self.same_data

    @cached_property
    def changed_df(self) -> DataFrame:
        """The DataFrame containing all rows that were found in both DataFrames but are not equal"""
        return self.diff_df.filter(PREDICATES.present_in_both & PREDICATES.row_changed).drop(
            EXISTS_COL_NAME, IS_EQUAL_COL_NAME
        )

    def _compute_diff_stats(self) -> DiffStats:
        """Given a diff_df and its list of join_cols, return stats about the number of differing or missing rows

        >>> from spark_frame.data_diff.package import _get_test_diff_df
        >>> _diff_df = _get_test_diff_df()
        >>> _diff_df.show()
        +---+----------------+----------------+-------------+------------+
        | id|              c1|              c2|   __EXISTS__|__IS_EQUAL__|
        +---+----------------+----------------+-------------+------------+
        |  1|    {a, a, true}|    {1, 1, true}| {true, true}|        true|
        |  2|    {b, b, true}|   {2, 4, false}| {true, true}|       false|
        |  3|{c, null, false}|{3, null, false}|{true, false}|       false|
        |  4|{null, f, false}|{null, 3, false}|{false, true}|       false|
        +---+----------------+----------------+-------------+------------+
        <BLANKLINE>
        >>> schema_diff_result = SchemaDiffResult(same_schema=True, diff_str="",
        ...                                       nb_cols=0, left_schema_str="", right_schema_str="")
        >>> DiffResult(schema_diff_result, diff_df=_diff_df, join_cols=['id'])._compute_diff_stats()
        DiffStats(total=4, no_change=1, changed=1, in_left=3, in_right=3, only_in_left=1, only_in_right=1)
        """
        res_df = self.diff_df.select(
            f.count(f.lit(1)).alias("total"),
            f.sum(f.when(PREDICATES.present_in_both & PREDICATES.row_is_equal, f.lit(1)).otherwise(f.lit(0))).alias(
                "no_change"
            ),
            f.sum(f.when(PREDICATES.present_in_both & PREDICATES.row_changed, f.lit(1)).otherwise(f.lit(0))).alias(
                "changed"
            ),
            f.sum(f.when(PREDICATES.in_left, f.lit(1)).otherwise(f.lit(0))).alias("in_left"),
            f.sum(f.when(PREDICATES.in_right, f.lit(1)).otherwise(f.lit(0))).alias("in_right"),
            f.sum(f.when(PREDICATES.only_in_left, f.lit(1)).otherwise(f.lit(0))).alias("only_in_left"),
            f.sum(f.when(PREDICATES.only_in_right, f.lit(1)).otherwise(f.lit(0))).alias("only_in_right"),
        )
        res = res_df.collect()
        return DiffStats(**{k: (v if v is not None else 0) for k, v in res[0].asDict().items()})

    def display(self, show_examples: bool = False):
        from spark_frame.data_diff.diff_result_analyzer import DiffResultAnalyzer

        self.schema_diff_result.display()
        analyzer = DiffResultAnalyzer(self.diff_format_options)
        analyzer.display_diff_results(self, show_examples)

    def export_to_html(self):
        from spark_frame.data_diff.diff_result_analyzer import DiffResultAnalyzer

        analyzer = DiffResultAnalyzer(self.diff_format_options)
        diff_result_summary = analyzer.get_diff_result_summary(self)
        export_html_diff_report(diff_result_summary)


def _get_test_diff_result() -> "DiffResult":
    from spark_frame.data_diff.package import _get_test_diff_df

    _diff_df = _get_test_diff_df()
    schema_diff_result = SchemaDiffResult(
        same_schema=True, diff_str="", nb_cols=0, left_schema_str="", right_schema_str=""
    )
    return DiffResult(schema_diff_result, diff_df=_diff_df, join_cols=["id"])
