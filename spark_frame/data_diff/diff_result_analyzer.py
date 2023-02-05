from typing import List, Optional, cast

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as f
from pyspark.sql.types import StructType

from spark_frame import transformations as df_transformations
from spark_frame.data_diff.diff_format_options import DiffFormatOptions
from spark_frame.data_diff.diff_result_summary import DiffResultSummary
from spark_frame.data_diff.diff_results import DiffResult
from spark_frame.data_diff.diff_stats import print_diff_stats
from spark_frame.data_diff.package import (
    CHANGED_COL_NAME,
    EXISTS_COL_NAME,
    IS_EQUAL_COL_NAME,
    PREDICATES,
    STRUCT_SEPARATOR_REPLACEMENT,
    canonize_col,
)
from spark_frame.utils import MAX_JAVA_INT, quote, quote_columns


class DiffResultAnalyzer:
    def __init__(self, diff_format_options: DiffFormatOptions = DiffFormatOptions()):
        self.diff_format_options = diff_format_options

    @staticmethod
    def _unpivot(diff_df: DataFrame, join_cols: List[str]):
        """Given a diff_df, builds an unpivoted version of it.
        All the values must be cast to STRING to make sure everything fits in the same column.

        Examples:
            >>> from spark_frame.data_diff.diff_result_analyzer import _get_test_intersection_diff_df
            >>> diff_df = _get_test_intersection_diff_df()
            >>> diff_df.show()
            +---+-------------+-------------+
            | id|           c1|           c2|
            +---+-------------+-------------+
            |  1|{a, d, false}| {1, 1, true}|
            |  2|{b, a, false}|{2, 4, false}|
            +---+-------------+-------------+
            <BLANKLINE>
            >>> DiffResultAnalyzer._unpivot(diff_df, join_cols=['id']).orderBy('id', 'column_name').show()
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

    def _format_diff_df(self, join_cols: List[str], diff_df: DataFrame) -> DataFrame:
        """Given a diff DataFrame, rename the columns to prefix them with the left_df_alias and right_df_alias."""
        return diff_df.select(
            *quote_columns(join_cols),
            *[
                col
                for col_name in diff_df.columns
                if col_name not in join_cols
                for col in [
                    diff_df[quote(col_name)]["left_value"].alias(
                        f"{self.diff_format_options.left_df_alias}__{col_name}"
                    ),
                    diff_df[quote(col_name)]["right_value"].alias(
                        f"{self.diff_format_options.right_df_alias}__{col_name}"
                    ),
                ]
            ],
        )

    def _display_diff_examples(self, diff_df: DataFrame, diff_per_col_df: DataFrame, join_cols: List[str]):
        """For each column that has differences, print examples of rows where such a difference occurs.

        Examples:
            >>> from spark_frame.data_diff.diff_results import _get_test_diff_result
            >>> diff_result = _get_test_diff_result()
            >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
            >>> top_per_col_state_df = analyzer._get_top_per_col_state_df(diff_result.diff_df, join_cols=['id'])
            >>> diff_per_col_df = analyzer._get_diff_per_col_df(top_per_col_state_df, diff_result)
            >>> diff_df = diff_result.diff_df
            >>> diff_df.show()
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
            >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
            >>> diff_per_col_df = _get_test_diff_per_col_df()
            >>> analyzer._display_diff_examples(diff_df, diff_per_col_df, join_cols = ['id'])
            Detailed examples :
            'c2' : 3 rows
            +---+----------+---------+
            | id|before__c2|after__c2|
            +---+----------+---------+
            |  2|         2|        3|
            |  3|         2|        4|
            |  4|         2|        4|
            +---+----------+---------+
            <BLANKLINE>
        """
        rows = (
            diff_per_col_df.where(~f.col("column_name").isin(join_cols))
            .where(f.col(f"counts.{CHANGED_COL_NAME}") > 0)
            .select("column_name", f.col(f"counts.{CHANGED_COL_NAME}").alias("total_nb_differences"))
            .collect()
        )
        diff_count_per_col = [(r[0], r[1]) for r in rows]
        print("Detailed examples :")
        for col, nb in diff_count_per_col:
            print(f"'{col}' : {nb} rows")
            rows_that_changed_for_that_column = (
                diff_df.where(PREDICATES.present_in_both)
                .where(~diff_df[quote(col)]["is_equal"])
                .select(*join_cols, *[quote(r[0]) for r in rows])
            )
            self._format_diff_df(join_cols, rows_that_changed_for_that_column).show(
                self.diff_format_options.nb_diffed_rows
            )

    @staticmethod
    def _get_top_per_col_state_df(diff_df: DataFrame, join_cols: List[str]) -> DataFrame:
        """Given a diff_df and its list of join_cols, return a DataFrame with the following properties:

        - One row per tuple (column_name, state, left_value, right_value)
          (where `state` can take the following values: "only_in_left", "only_in_right", "no_change", "changed")
        - A column `nb` that gives the number of occurrence of this specific tuple
        - At most `max_nb_rows_per_col_state` per tuple (column_name, state). Rows with the highest "nb" are kept first.
        - A column `col_state_total` that gives the corresponding sum for the tuple (column_name, state)
          before filtering the rows

        Args:
            diff_df: A diff_df
            join_cols: The list of columns used for the join

        Returns:
            A Dataframe

        Examples:
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
            >>> (DiffResultAnalyzer._get_top_per_col_state_df(_diff_df, join_cols = ['id'])
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
            >>> (DiffResultAnalyzer._get_top_per_col_state_df(_diff_df, join_cols = ['id'])
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
        for join_col in join_cols:
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
        unpivoted_diff_df = DiffResultAnalyzer._unpivot(diff_df.drop(IS_EQUAL_COL_NAME), [EXISTS_COL_NAME])
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

    def _get_diff_per_col_df(
        self, top_per_col_state_df: DataFrame, diff_result: DiffResult, max_nb_rows_per_col_state: Optional[int] = None
    ) -> DataFrame:
        """Given a top_per_col_state_df, return a Dict[str, int] that gives for each column and each
        column state (changed, no_change, only_in_left, only_in_right) the total number of occurences
        and the most frequent occurrences.

        !!! warning
            The arrays contained in the field `diff` are NOT guaranteed to be sorted,
            and Spark currently does not provide any way to perform a sort_by on an ARRAY<STRUCT>.

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
            >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
            >>> top_per_col_state_df = analyzer._get_top_per_col_state_df(diff_result.diff_df, join_cols = ['id'])
            >>> top_per_col_state_df.show()
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

            >>> df = analyzer._get_diff_per_col_df(top_per_col_state_df, diff_result)
            >>> from spark_frame import nested
            >>> nested.print_schema(df)
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
             |-- diff.no_change!.left_value: string (nullable = true)
             |-- diff.no_change!.right_value: string (nullable = true)
             |-- diff.no_change!.nb: long (nullable = false)
             |-- diff.only_in_left!.left_value: string (nullable = true)
             |-- diff.only_in_left!.right_value: string (nullable = true)
             |-- diff.only_in_left!.nb: long (nullable = false)
             |-- diff.only_in_right!.left_value: string (nullable = true)
             |-- diff.only_in_right!.right_value: string (nullable = true)
             |-- diff.only_in_right!.nb: long (nullable = false)
            <BLANKLINE>
            >>> df.show(truncate=False)  # noqa: E501
            +-------------+-----------+---------------+----------------------------------------------------------------------------------+
            |column_number|column_name|counts         |diff                                                                              |
            +-------------+-----------+---------------+----------------------------------------------------------------------------------+
            |0            |id         |{6, 0, 4, 1, 1}|{[], [{1, 1, 1}, {2, 2, 1}, {3, 3, 1}, {4, 4, 1}], [{5, null, 1}], [{null, 6, 1}]}|
            |1            |c1         |{6, 0, 4, 1, 1}|{[], [{b, b, 3}, {a, a, 1}], [{c, null, 1}], [{null, f, 1}]}                      |
            |2            |c2         |{6, 3, 1, 1, 1}|{[{2, 4, 2}, {2, 3, 1}], [{1, 1, 1}], [{3, null, 1}], [{null, 3, 1}]}             |
            +-------------+-----------+---------------+----------------------------------------------------------------------------------+
            <BLANKLINE>
            >>> analyzer._get_diff_per_col_df(top_per_col_state_df.orderBy("nb"), diff_result).show(truncate=False)  # noqa: E501
            +-------------+-----------+---------------+----------------------------------------------------------------------------------+
            |column_number|column_name|counts         |diff                                                                              |
            +-------------+-----------+---------------+----------------------------------------------------------------------------------+
            |0            |id         |{6, 0, 4, 1, 1}|{[], [{1, 1, 1}, {2, 2, 1}, {3, 3, 1}, {4, 4, 1}], [{5, null, 1}], [{null, 6, 1}]}|
            |1            |c1         |{6, 0, 4, 1, 1}|{[], [{a, a, 1}, {b, b, 3}], [{c, null, 1}], [{null, f, 1}]}                      |
            |2            |c2         |{6, 3, 1, 1, 1}|{[{2, 3, 1}, {2, 4, 2}], [{1, 1, 1}], [{3, null, 1}], [{null, 3, 1}]}             |
            +-------------+-----------+---------------+----------------------------------------------------------------------------------+
            <BLANKLINE>
        """
        if max_nb_rows_per_col_state is None:
            _max_nb_rows_per_col_state = self.diff_format_options.nb_diffed_rows
        else:
            _max_nb_rows_per_col_state = max_nb_rows_per_col_state

        diff_stats = diff_result.diff_stats
        columns = diff_result.diff_df.columns[:-2]
        col_df = diff_result.diff_df.sparkSession.createDataFrame(
            list(enumerate(columns)), "column_number INT, column_name STRING"
        ).withColumn("column_name", f.regexp_replace(f.col("column_name"), STRUCT_SEPARATOR_REPLACEMENT, "."))
        # col_df.show()
        # +-------------+-----------+
        # |column_number|column_name|
        # +-------------+-----------+
        # |            0|         id|
        # |            1|         c1|
        # |            2|         c2|
        # +-------------+-----------+

        # top_per_col_state_df.show()
        # +-----------+-------------+----------+-----------+---+---------------+
        # |column_name|        state|left_value|right_value| nb|col_state_total|
        # +-----------+-------------+----------+-----------+---+---------------+
        # |         c1|    no_change|         a|          a|  1|              2|
        # |         c1|    no_change|         b|          b|  1|              2|
        # |         c1| only_in_left|         c|       null|  1|              1|
        # |         c1|only_in_right|      null|          f|  1|              1|
        # |         c2|      changed|         2|          4|  1|              1|
        # |         c2|    no_change|         1|          1|  1|              1|
        # |         c2| only_in_left|         3|       null|  1|              1|
        # |         c2|only_in_right|      null|          3|  1|              1|
        # |         id|    no_change|         1|          1|  1|              2|
        # |         id|    no_change|         2|          2|  1|              2|
        # |         id| only_in_left|         3|       null|  1|              1|
        # |         id|only_in_right|      null|          4|  1|              1|
        # +-----------+-------------+----------+-----------+---+---------------+
        pivoted_df = (
            top_per_col_state_df.groupBy("column_name")
            .pivot("state", values=["changed", "no_change", "only_in_left", "only_in_right"])
            .agg(
                f.sum("nb").alias("nb"),
                f.expr(
                    f"""
                ARRAY_AGG(
                    CASE WHEN row_num <= {_max_nb_rows_per_col_state} THEN STRUCT(left_value, right_value, nb) END
                )"""
                ).alias("diff"),
            )
        )
        # pivoted_df.show(truncate=False)
        # +-----------+----------+------------+--------+----------------------+---------------+-----------------+----------------+------------------+  # noqa: E501
        # |column_name|changed_nb|changed_diff|equal_nb|equal_diff            |only_in_left_nb|only_in_left_diff|only_in_right_nb|only_in_right_diff|  # noqa: E501
        # +-----------+----------+------------+--------+----------------------+---------------+-----------------+----------------+------------------+  # noqa: E501
        # |c1         |null      |[]          |2       |[{a, a, 1}, {b, b, 1}]|1              |[{c, null, 1}]   |1               |[{null, f, 1}]    |  # noqa: E501
        # |c2         |1         |[{2, 4, 1}] |1       |[{1, 1, 1}]           |1              |[{3, null, 1}]   |1               |[{null, 3, 1}]    |  # noqa: E501
        # |id         |null      |[]          |2       |[{1, 1, 1}, {2, 2, 1}]|1              |[{3, null, 1}]   |1               |[{null, 4, 1}]    |  # noqa: E501
        # +-----------+----------+------------+--------+----------------------+---------------+-----------------+----------------+------------------+  # noqa: E501
        df = (
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
        df = df.select(
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
        # df.show(truncate=False)
        # +-----------+------------------+------------------------------------------------------------+
        # |column_name|counts            |diff                                                        |
        # +-----------+------------------+------------------------------------------------------------+
        # |id         |{4, 0, 2, 0, 0, 0}|{[], [], [], []}                                            |
        # |c1         |{4, 0, 2, 1, 1, 1}|{[], [{a, a, 1}, {b, b, 1}], [{c, null, 1}], [{null, f, 1}]}|
        # |c2         |{4, 1, 1, 1, 1, 1}|{[{2, 4, 1}], [{1, 1, 1}], [{3, null, 1}], [{null, 3, 1}]}  |
        # +-----------+------------------+------------------------------------------------------------+
        return df

    @staticmethod
    def _display_changed(diff_per_col_df: DataFrame):
        """Displays the results of the diff analysis.

        We first display a summary of all columns that changed with the number of changes,
        then for each column, we display a summary of the most frequent changes and then
        we display examples of rows where this column changed, along with all the other columns
        that changed in this diff.

        Example:

        >>> diff_per_col_df = _get_test_diff_per_col_df()
        >>> diff_per_col_df.show(truncate=False)  # noqa: E501
        +-------------+-----------+---------------+----------------------------------------------------------------------------------+
        |column_number|column_name|counts         |diff                                                                              |
        +-------------+-----------+---------------+----------------------------------------------------------------------------------+
        |0            |id         |{6, 0, 4, 1, 1}|{[], [{1, 1, 1}, {2, 2, 1}, {3, 3, 1}, {4, 4, 1}], [{5, null, 1}], [{null, 6, 1}]}|
        |1            |c1         |{6, 0, 4, 1, 1}|{[], [{a, a, 1}, {b, b, 3}], [{c, null, 1}], [{null, f, 1}]}                      |
        |2            |c2         |{6, 3, 1, 1, 1}|{[{2, 3, 1}, {2, 4, 2}], [{1, 1, 1}], [{3, null, 1}], [{null, 3, 1}]}             |
        +-------------+-----------+---------------+----------------------------------------------------------------------------------+
        <BLANKLINE>
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
         |-- diff.no_change!.left_value: string (nullable = true)
         |-- diff.no_change!.right_value: string (nullable = true)
         |-- diff.no_change!.nb: long (nullable = false)
         |-- diff.only_in_left!.left_value: string (nullable = true)
         |-- diff.only_in_left!.right_value: string (nullable = true)
         |-- diff.only_in_left!.nb: long (nullable = false)
         |-- diff.only_in_right!.left_value: string (nullable = true)
         |-- diff.only_in_right!.right_value: string (nullable = true)
         |-- diff.only_in_right!.nb: long (nullable = false)
        <BLANKLINE>
        >>> DiffResultAnalyzer._display_changed(diff_per_col_df)
        +-----------+-------------+----------+-----------+--------------+
        |column_name|total_nb_diff|left_value|right_value|nb_differences|
        +-----------+-------------+----------+-----------+--------------+
        |c2         |3            |2         |4          |2             |
        |c2         |3            |2         |3          |1             |
        +-----------+-------------+----------+-----------+--------------+
        <BLANKLINE>
        """
        df = diff_per_col_df.where(f.col(f"counts.{CHANGED_COL_NAME}") > 0)
        df = df.select(
            "column_name",
            f.col(f"counts.{CHANGED_COL_NAME}").alias("total_nb_diff"),
            f.explode("diff.changed").alias("diff"),
        ).orderBy("column_number", f.desc("diff.nb"))
        df = df.select(
            "column_name",
            "total_nb_diff",
            "diff.left_value",
            "diff.right_value",
            f.col("diff.nb").alias("nb_differences"),
        )
        df.show(MAX_JAVA_INT, truncate=False)

    @staticmethod
    def _display_only_in_left_or_right(diff_per_col_df: DataFrame, left_or_right: str) -> None:
        """Displays the results of the diff analysis.

        We first display a summary of all columns that changed with the number of changes,
        then for each column, we display a summary of the most frequent changes and then
        we display examples of rows where this column changed, along with all the other columns
        that changed in this diff.

        Example:

        >>> diff_per_col_df = _get_test_diff_per_col_df()
        >>> diff_per_col_df.show(truncate=False)  # noqa: E501
        +-------------+-----------+---------------+----------------------------------------------------------------------------------+
        |column_number|column_name|counts         |diff                                                                              |
        +-------------+-----------+---------------+----------------------------------------------------------------------------------+
        |0            |id         |{6, 0, 4, 1, 1}|{[], [{1, 1, 1}, {2, 2, 1}, {3, 3, 1}, {4, 4, 1}], [{5, null, 1}], [{null, 6, 1}]}|
        |1            |c1         |{6, 0, 4, 1, 1}|{[], [{a, a, 1}, {b, b, 3}], [{c, null, 1}], [{null, f, 1}]}                      |
        |2            |c2         |{6, 3, 1, 1, 1}|{[{2, 3, 1}, {2, 4, 2}], [{1, 1, 1}], [{3, null, 1}], [{null, 3, 1}]}             |
        +-------------+-----------+---------------+----------------------------------------------------------------------------------+
        <BLANKLINE>
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
         |-- diff.no_change!.left_value: string (nullable = true)
         |-- diff.no_change!.right_value: string (nullable = true)
         |-- diff.no_change!.nb: long (nullable = false)
         |-- diff.only_in_left!.left_value: string (nullable = true)
         |-- diff.only_in_left!.right_value: string (nullable = true)
         |-- diff.only_in_left!.nb: long (nullable = false)
         |-- diff.only_in_right!.left_value: string (nullable = true)
         |-- diff.only_in_right!.right_value: string (nullable = true)
         |-- diff.only_in_right!.nb: long (nullable = false)
        <BLANKLINE>
        >>> DiffResultAnalyzer._display_only_in_left_or_right(diff_per_col_df, "left")
        +-----------+-----+---+
        |column_name|value|nb |
        +-----------+-----+---+
        |id         |5    |1  |
        |c1         |c    |1  |
        |c2         |3    |1  |
        +-----------+-----+---+
        <BLANKLINE>
        >>> DiffResultAnalyzer._display_only_in_left_or_right(diff_per_col_df, "right")
        +-----------+-----+---+
        |column_name|value|nb |
        +-----------+-----+---+
        |id         |6    |1  |
        |c1         |f    |1  |
        |c2         |3    |1  |
        +-----------+-----+---+
        <BLANKLINE>

        """
        df = diff_per_col_df.select("column_name", f.explode(f"diff.only_in_{left_or_right}").alias("diff"))
        df = df.select("column_name", f.col(f"diff.{left_or_right}_value").alias("value"), f.col("diff.nb").alias("nb"))
        df.show(MAX_JAVA_INT, truncate=False)

    def display_diff_results(self, diff_result: DiffResult, show_examples: bool):
        join_cols = diff_result.join_cols
        top_per_col_state_df = DiffResultAnalyzer._get_top_per_col_state_df(
            diff_result.diff_df, join_cols
        ).localCheckpoint()
        diff_per_col_df = self._get_diff_per_col_df(top_per_col_state_df, diff_result).localCheckpoint()

        left_df_alias = self.diff_format_options.left_df_alias
        right_df_alias = self.diff_format_options.right_df_alias
        diff_stats = diff_result.diff_stats

        print_diff_stats(diff_stats, left_df_alias, right_df_alias)

        if diff_stats.changed > 0:
            print("Found the following changes:")
            self._display_changed(diff_per_col_df)
            if show_examples:
                self._display_diff_examples(diff_result.diff_df, diff_per_col_df, join_cols)
        if diff_stats.only_in_left > 0:
            print(f"{diff_stats.only_in_left} rows were only found in '{left_df_alias}' :")
            print(f"Most frequent values in '{left_df_alias}' for each column :")
            self._display_only_in_left_or_right(diff_per_col_df, "left")
        if diff_stats.only_in_right > 0:
            print(f"{diff_stats.only_in_left} rows were only found in '{right_df_alias}' :")
            print(f"Most frequent values in '{right_df_alias}' for each column :")
            self._display_only_in_left_or_right(diff_per_col_df, "right")

    def get_diff_result_summary(self, diff_result: DiffResult) -> DiffResultSummary:
        join_cols = diff_result.join_cols
        top_per_col_state_df = DiffResultAnalyzer._get_top_per_col_state_df(
            diff_result.diff_df, join_cols
        ).localCheckpoint()
        diff_per_col_df = self._get_diff_per_col_df(top_per_col_state_df.orderBy("nb"), diff_result).localCheckpoint()
        summary = DiffResultSummary(
            left_df_alias=self.diff_format_options.left_df_alias,
            right_df_alias=self.diff_format_options.right_df_alias,
            diff_per_col_df=diff_per_col_df,
            schema_diff_result=diff_result.schema_diff_result,
            join_cols=diff_result.join_cols,
            same_schema=diff_result.same_schema,
            same_data=diff_result.same_data,
            is_ok=diff_result.is_ok,
        )
        return summary


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


def _get_test_diff_per_col_df() -> DataFrame:
    """Return an example of diff_per_col_df for testing purposes.
    We intentionally sort top_per_col_state_df by increasing "nb" to simulate the fact that we don't have
    any way to guarantee that the diff arrays will be sorted by decreasing order of "nb" in the `diff` column.
    """
    from spark_frame.data_diff.diff_results import _get_test_diff_result

    diff_result = _get_test_diff_result()
    analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
    top_per_col_state_df = DiffResultAnalyzer._get_top_per_col_state_df(diff_result.diff_df, join_cols=["id"])
    df = analyzer._get_diff_per_col_df(top_per_col_state_df.orderBy("nb"), diff_result)
    return df
