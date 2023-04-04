from functools import cached_property, lru_cache
from typing import List, Optional, cast

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f
from pyspark.sql.types import StructType

from spark_frame import transformations as df_transformations
from spark_frame.data_diff.diff_format_options import DiffFormatOptions
from spark_frame.data_diff.diff_stats import DiffStats
from spark_frame.data_diff.export import (
    DEFAULT_HTML_REPORT_ENCODING,
    DEFAULT_HTML_REPORT_OUTPUT_FILE_PATH,
    export_html_diff_report,
)
from spark_frame.data_diff.package import (
    EXISTS_COL_NAME,
    IS_EQUAL_COL_NAME,
    PREDICATES,
    STRUCT_SEPARATOR_REPLACEMENT,
    canonize_col,
)
from spark_frame.data_diff.schema_diff import SchemaDiffResult
from spark_frame.utils import quote, quote_columns


def _unpivot(diff_df: DataFrame, join_cols: List[str]) -> DataFrame:
    """Given a diff_df, builds an unpivoted version of it.
    All the values must be cast to STRING to make sure everything fits in the same column.

    Examples:
        >>> from spark_frame.data_diff.diff_results import _get_test_intersection_diff_df
        >>> diff_df = _get_test_intersection_diff_df()
        >>> diff_df.show()
        +---+-------------+-------------+
        | id|           c1|           c2|
        +---+-------------+-------------+
        |  1|{a, d, false}| {1, 1, true}|
        |  2|{b, a, false}|{2, 4, false}|
        +---+-------------+-------------+
        <BLANKLINE>
        >>> _unpivot(diff_df, join_cols=['id']).orderBy('id', 'column_name').show()
        +---+-----------+-------------+
        | id|column_name|         diff|
        +---+-----------+-------------+
        |  1|         c1|{a, d, false}|
        |  1|         c2| {1, 1, true}|
        |  2|         c1|{b, a, false}|
        |  2|         c2|{2, 4, false}|
        +---+-----------+-------------+
        <BLANKLINE>
    """

    diff_df = diff_df.select(
        *quote_columns(join_cols),
        *[
            f.struct(
                canonize_col(diff_df[field.name + ".left_value"], cast(StructType, field.dataType).fields[0])
                .cast("STRING")
                .alias("left_value"),
                canonize_col(diff_df[field.name + ".right_value"], cast(StructType, field.dataType).fields[0])
                .cast("STRING")
                .alias("right_value"),
                diff_df[quote(field.name) + ".is_equal"].alias("is_equal"),
            ).alias(field.name)
            for field in diff_df.schema.fields
            if field.name not in join_cols
        ],
    )

    unpivoted_df = df_transformations.unpivot(
        diff_df, pivot_columns=join_cols, key_alias="column_name", value_alias="diff"
    )
    unpivoted_df = unpivoted_df.withColumn(
        "column_name", f.regexp_replace(f.col("column_name"), STRUCT_SEPARATOR_REPLACEMENT, ".")
    )
    return unpivoted_df


class DiffResult:
    def __init__(
        self,
        schema_diff_result: SchemaDiffResult,
        diff_df: DataFrame,
        common_cols: List[str],
        join_cols: List[str],
    ) -> None:
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
        self.common_cols = common_cols
        """The list of columns present in both DataFrames"""
        self.join_cols = join_cols
        """The list of column names to join"""
        self._changed_df_shards: Optional[List[DataFrame]] = None

    @property
    def same_schema(self) -> bool:
        return self.schema_diff_result.same_schema

    @property
    def same_data(self) -> bool:
        return self.diff_stats.same_data

    @property
    def is_ok(self) -> bool:
        return self.same_schema and self.same_data

    @cached_property
    def diff_stats(self) -> DiffStats:
        return self._compute_diff_stats()

    @cached_property
    def top_per_col_state_df(self) -> DataFrame:
        return self._compute_top_per_col_state_df().localCheckpoint()

    @cached_property
    def changed_df(self) -> DataFrame:
        """The DataFrame containing all rows that were found in both DataFrames but are not equal"""
        return self.diff_df.filter(PREDICATES.present_in_both & PREDICATES.row_changed).drop(
            EXISTS_COL_NAME, IS_EQUAL_COL_NAME
        )

    @lru_cache()
    def get_diff_per_col_df(self, max_nb_rows_per_col_state: int) -> DataFrame:
        """Return a Dict[str, int] that gives for each column and each column state (changed, no_change, only_in_left,
        only_in_right) the total number of occurences and the most frequent occurrences.

        The results returned by this method are cached to avoid unecessary recomputations.

        !!! warning
            The arrays contained in the field `diff` are NOT guaranteed to be sorted,
            and Spark currently does not provide any way to perform a sort_by on an ARRAY<STRUCT>.

        Args:
            max_nb_rows_per_col_state: The maximal size of the arrays in `diff`

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

            >>> diff_per_col_df = diff_result.get_diff_per_col_df(max_nb_rows_per_col_state=10)
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
        """
        from spark_frame.data_diff.diff_per_col import _get_diff_per_col_df

        return _get_diff_per_col_df(
            self, max_nb_rows_per_col_state=max_nb_rows_per_col_state, top_per_col_state_df=None
        ).localCheckpoint()

    def _compute_diff_stats(self) -> DiffStats:
        """Given a diff_df and its list of join_cols, return stats about the number of differing or missing rows

        >>> from spark_frame.data_diff.package import _get_test_diff_df
        >>> _diff_df = _get_test_diff_df()
        >>> _diff_df.show()
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
        >>> schema_diff_result = SchemaDiffResult(same_schema=True, diff_str="",
        ...                                       nb_cols=0, left_schema_str="", right_schema_str="")
        >>> DiffResult(schema_diff_result, diff_df=_diff_df,
        ...            common_cols=["id", "c1", "c2"] , join_cols=['id'])._compute_diff_stats()
        DiffStats(total=6, no_change=1, changed=3, in_left=5, in_right=5, only_in_left=1, only_in_right=1)
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

    def _compute_top_per_col_state_df(self) -> DataFrame:
        """Given a diff_df and its list of join_cols, return a DataFrame with the following properties:

        - One row per tuple (column_name, state, left_value, right_value)
          (where `state` can take the following values: "only_in_left", "only_in_right", "no_change", "changed")
        - A column `nb` that gives the number of occurrence of this specific tuple
        - At most `max_nb_rows_per_col_state` per tuple (column_name, state). Rows with the highest "nb" are kept first.
        - A column `col_state_total` that gives the corresponding sum for the tuple (column_name, state)
          before filtering the rows

        Returns:
            A Dataframe

        Examples:
            >>> from spark_frame.data_diff.diff_results import _get_test_diff_result
            >>> _diff_result = _get_test_diff_result()
            >>> _diff_result.diff_df.show()
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
            >>> (_diff_result._compute_top_per_col_state_df()
            ...  .orderBy("column_name", "state", "left_value")
            ... ).show()
            +-----------+-------------+----------+-----------+---+---------------+-------+
            |column_name|        state|left_value|right_value| nb|col_state_total|row_num|
            +-----------+-------------+----------+-----------+---+---------------+-------+
            |         c1|    no_change|         a|          a|  1|              4|      2|
            |         c1|    no_change|         b|          b|  3|              4|      1|
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

            *With `max_nb_rows_per_col_state=1`*
            >>> (_diff_result._compute_top_per_col_state_df()
            ...  .orderBy("column_name", "state", "left_value")
            ... ).show()
            +-----------+-------------+----------+-----------+---+---------------+-------+
            |column_name|        state|left_value|right_value| nb|col_state_total|row_num|
            +-----------+-------------+----------+-----------+---+---------------+-------+
            |         c1|    no_change|         a|          a|  1|              4|      2|
            |         c1|    no_change|         b|          b|  3|              4|      1|
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
        """
        diff_df = self.diff_df
        for join_col in self.join_cols:
            diff_df = diff_df.withColumn(
                join_col,
                f.when(
                    PREDICATES.only_in_left,
                    f.struct(
                        f.col(join_col).alias("left_value"),
                        f.lit(None).alias("right_value"),
                        f.lit(False).alias("is_equal"),
                    ),
                )
                .when(
                    PREDICATES.only_in_right,
                    f.struct(
                        f.lit(None).alias("left_value"),
                        f.col(join_col).alias("right_value"),
                        f.lit(False).alias("is_equal"),
                    ),
                )
                .otherwise(
                    f.struct(
                        f.col(join_col).alias("left_value"),
                        f.col(join_col).alias("right_value"),
                        f.lit(True).alias("is_equal"),
                    )
                ),
            )
        unpivoted_diff_df = _unpivot(diff_df.drop(IS_EQUAL_COL_NAME), [EXISTS_COL_NAME])
        df_2 = unpivoted_diff_df.select(
            "column_name",
            f.when(PREDICATES.only_in_left, f.lit("only_in_left"))
            .when(PREDICATES.only_in_right, f.lit("only_in_right"))
            .when(f.col("diff")["is_equal"], f.lit("no_change"))
            .otherwise(f.lit("changed"))
            .alias("state"),
            "diff.left_value",
            "diff.right_value",
        )
        window = Window.partitionBy("column_name", "state").orderBy(
            f.col("nb").desc(), f.col("left_value"), f.col("right_value")
        )
        df = (
            df_2.groupBy("column_name", "state", "left_value", "right_value")
            .agg(f.count(f.lit(1)).alias("nb"))
            .withColumn("col_state_total", f.sum("nb").over(Window.partitionBy("column_name", "state")))
            .withColumn("row_num", f.row_number().over(window))
        )
        return df

    def display(
        self, show_examples: bool = False, diff_format_options: DiffFormatOptions = DiffFormatOptions()
    ) -> None:
        """Print a summary of the results in the standard output

        Args:
            show_examples: If true, display example of rows for each type of change
            diff_format_options: Formatting options
        """
        from spark_frame.data_diff.diff_result_analyzer import DiffResultAnalyzer

        self.schema_diff_result.display()
        analyzer = DiffResultAnalyzer(diff_format_options)
        analyzer.display_diff_results(self, show_examples)

    def export_to_html(
        self,
        title: Optional[str] = None,
        output_file_path: str = DEFAULT_HTML_REPORT_OUTPUT_FILE_PATH,
        encoding: str = DEFAULT_HTML_REPORT_ENCODING,
        diff_format_options: DiffFormatOptions = DiffFormatOptions(),
    ) -> None:
        """Generate an HTML report of this diff result.

        This generates a file named diff_report.html in the current working directory.
        It can be open directly with a web browser.

        Args:
            title: The title of the report
            encoding: Encoding used when writing the html report
            output_file_path: Path of the file to write to
            diff_format_options: Formatting options
        """
        from spark_frame.data_diff.diff_result_analyzer import DiffResultAnalyzer

        analyzer = DiffResultAnalyzer(diff_format_options)
        diff_result_summary = analyzer.get_diff_result_summary(self)
        export_html_diff_report(diff_result_summary, title=title, output_file_path=output_file_path, encoding=encoding)


def _get_test_diff_result() -> "DiffResult":
    from spark_frame.data_diff.package import _get_test_diff_df

    _diff_df = _get_test_diff_df()
    schema_diff_result = SchemaDiffResult(
        same_schema=True, diff_str="", nb_cols=0, left_schema_str="", right_schema_str=""
    )
    return DiffResult(schema_diff_result, diff_df=_diff_df, common_cols=["id", "c1", "c2"], join_cols=["id"])


def _get_test_intersection_diff_df() -> DataFrame:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("doctest").getOrCreate()
    diff_df = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(
                1 as id,
                STRUCT("a" as left_value, "d" as right_value, False as is_equal) as c1,
                STRUCT(1 as left_value, 1 as right_value, True as is_equal) as c2
            ),
            STRUCT(
                2 as id,
                STRUCT("b" as left_value, "a" as right_value, False as is_equal) as c1,
                STRUCT(2 as left_value, 4 as right_value, False as is_equal) as c2
            )
        ))
    """
    )
    return diff_df
