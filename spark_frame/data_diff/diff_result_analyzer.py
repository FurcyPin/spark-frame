from typing import List, Optional, cast

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as f
from pyspark.sql.types import StructType

from spark_frame import transformations as df_transformations
from spark_frame.data_diff.diff_format_options import DiffFormatOptions
from spark_frame.data_diff.diff_result_summary import DiffResultSummary
from spark_frame.data_diff.diff_results import DiffResult
from spark_frame.data_diff.diff_stats import print_diff_stats
from spark_frame.data_diff.package import (
    EXISTS_COL_NAME,
    IS_EQUAL_COL_NAME,
    PREDICATES,
    STRUCT_SEPARATOR_REPLACEMENT,
    canonize_col,
)
from spark_frame.transformations import analyze
from spark_frame.utils import assert_true, quote, quote_columns


class DiffResultAnalyzer:
    def __init__(self, diff_format_options: DiffFormatOptions = DiffFormatOptions()):
        self.diff_format_options = diff_format_options

    def _unpivot(self, diff_df: DataFrame, join_cols: List[str]):
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
            >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
            >>> analyzer._unpivot(diff_df, join_cols=['id']).orderBy('id', 'column_name').show()
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

        def truncate_string(col: Column) -> Column:
            return f.when(
                f.length(col) > f.lit(self.diff_format_options.max_string_length),
                f.concat(f.substring(col, 0, self.diff_format_options.max_string_length - 3), f.lit("...")),
            ).otherwise(col)

        diff_df = diff_df.select(
            *quote_columns(join_cols),
            *[
                f.struct(
                    truncate_string(
                        canonize_col(
                            diff_df[field.name + ".left_value"], cast(StructType, field.dataType).fields[0]
                        ).cast("STRING")
                    ).alias("left_value"),
                    truncate_string(
                        canonize_col(
                            diff_df[field.name + ".right_value"], cast(StructType, field.dataType).fields[0]
                        ).cast("STRING")
                    ).alias("right_value"),
                    diff_df[quote(field.name) + ".is_equal"].alias("is_equal"),
                ).alias(field.name)
                for field in diff_df.schema.fields
                if field.name not in join_cols
            ],
        )

        unpivot = df_transformations.unpivot(
            diff_df, pivot_columns=join_cols, key_alias="column_name", value_alias="diff"
        )
        return unpivot

    def _get_diff_per_col_df(self, diff_count_per_col_df: DataFrame, diff_result: DiffResult) -> DataFrame:
        """Given a diff_count_per_col_df, return a Dict[str, int] that gives for each column the total number
        of differences.

        Examples:
            >>> from spark_frame.data_diff.diff_results import _get_test_diff_result
            >>> diff_result = _get_test_diff_result()
            >>> diff_result.diff_df.show()
            +---+----------------+----------------+-------------+------------+
            | id|              c1|              c2|   __EXISTS__|__IS_EQUAL__|
            +---+----------------+----------------+-------------+------------+
            |  1|    {a, a, true}|    {1, 1, true}| {true, true}|        true|
            |  2|    {b, b, true}|   {2, 4, false}| {true, true}|       false|
            |  3|{c, null, false}|{3, null, false}|{true, false}|       false|
            |  4|{null, f, false}|{null, 3, false}|{false, true}|       false|
            +---+----------------+----------------+-------------+------------+
            <BLANKLINE>
            >>> diff_result.diff_stats
            DiffStats(total=4, no_change=1, changed=1, in_left=3, in_right=3, only_in_left=1, only_in_right=1)
            >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
            >>> diff_count_per_col_df = analyzer._get_diff_count_per_col_df(diff_result.changed_df, join_cols=["id"])
            >>> df = analyzer._get_diff_per_col_df(diff_count_per_col_df, diff_result)
            >>> df.show()
            +-----------+-----+---------+-------+------------+-------------+-----------+
            |column_name|total|no_change|changed|only_in_left|only_in_right|differences|
            +-----------+-----+---------+-------+------------+-------------+-----------+
            |         id|    4|        2|      0|           1|            1|         []|
            |         c1|    4|        2|      0|           1|            1|         []|
            |         c2|    4|        1|      1|           1|            1|[{2, 4, 1}]|
            +-----------+-----+---------+-------+------------+-------------+-----------+
            <BLANKLINE>
        """
        diff_stats = diff_result.diff_stats
        columns = diff_result.diff_df.columns[:-2]
        col_df = diff_result.diff_df.sparkSession.createDataFrame(
            list(enumerate(columns)), "column_number INT, column_name STRING"
        )
        agg_per_col_df = diff_count_per_col_df.groupBy("column_name").agg(
            f.max("total_nb_differences").alias("changed"),
            f.expr("ARRAY_AGG(STRUCT(left_value, right_value, nb_differences))").alias("differences"),
        )
        return (
            col_df.join(agg_per_col_df, "column_name", "left")
            .withColumn("changed", f.coalesce(f.col("changed"), f.lit(0)))
            .select(
                f.col("column_name").alias("column_name"),
                f.lit(diff_stats.total).alias("total"),
                (f.lit(diff_stats.in_both) - f.col("changed")).alias("no_change"),
                f.col("changed"),
                f.lit(diff_stats.only_in_left).alias("only_in_left"),
                f.lit(diff_stats.only_in_right).alias("only_in_right"),
                f.coalesce(f.col("differences"), f.array()).alias("differences"),
            )
        )

    def _get_diff_count_per_col_df(self, diff_df: DataFrame, join_cols: List[str]) -> DataFrame:
        """Given a diff_df and its list of join_cols, return a DataFrame with the following properties:

        - One row per "diff tuple" (col_name, col_value_left, col_value_right)
        - A column nb_differences that gives the number of occurrence of each "diff tuple"
        - A column total_nb_differences that gives total number of differences found for this col_name.

        Examples:
            >>> from spark_frame.data_diff.diff_result_analyzer import _get_test_intersection_diff_df
            >>> _diff_df = _get_test_intersection_diff_df()
            >>> _diff_df.show()
            +---+-------------+-------------+
            | id|           c1|           c2|
            +---+-------------+-------------+
            |  1|{a, d, false}| {1, 1, true}|
            |  2|{b, a, false}|{2, 4, false}|
            +---+-------------+-------------+
            <BLANKLINE>
            >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
            >>> diff_count_per_col_df = analyzer._get_diff_count_per_col_df(_diff_df, join_cols = ['id'])
            >>> diff_count_per_col_df.orderBy("column_name", "left_value").show()
            +-----------+----------+-----------+--------------+--------------------+
            |column_name|left_value|right_value|nb_differences|total_nb_differences|
            +-----------+----------+-----------+--------------+--------------------+
            |         c1|         a|          d|             1|                   2|
            |         c1|         b|          a|             1|                   2|
            |         c2|         2|          4|             1|                   1|
            +-----------+----------+-----------+--------------+--------------------+
            <BLANKLINE>
        """
        unpivoted_diff_df = self._unpivot(diff_df, join_cols)
        diff_count_per_col_df = self._build_diff_count_per_col_df_from_unpivoted_diff_df(unpivoted_diff_df).orderBy(
            "column_name"
        )
        return diff_count_per_col_df

    def _display_diff_count_per_col(self, diff_count_per_col_df: DataFrame):
        """Displays the results of the diff analysis.
        We first display the a summary of all columns that changed with the number of changes,
        then for each column, we display a summary of the most frequent changes and then
        we display examples of rows where this column changed, along with all the other columns
        that changed in this diff.

        Example:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> diff_count_per_col_df = spark.sql(# noqa: E501
        ... '''
        ...     SELECT INLINE(ARRAY(
        ...         STRUCT('c1' as column_name, 'a' as left_value, 'd' as right_value, 1 as nb_differences, 3 as total_nb_differences),
        ...         STRUCT('c3' as column_name, '2' as left_value, '4' as right_value, 1 as nb_differences, 1 as total_nb_differences)
        ... ))''')
        >>> diff_count_per_col_df.show()
        +-----------+----------+-----------+--------------+--------------------+
        |column_name|left_value|right_value|nb_differences|total_nb_differences|
        +-----------+----------+-----------+--------------+--------------------+
        |         c1|         a|          d|             1|                   3|
        |         c3|         2|          4|             1|                   1|
        +-----------+----------+-----------+--------------+--------------------+
        <BLANKLINE>
        >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
        >>> analyzer._display_diff_count_per_col(diff_count_per_col_df)
        Found the following differences:
        +-----------+-------------+------+-----+--------------+
        |column_name|total_nb_diff|before|after|nb_differences|
        +-----------+-------------+------+-----+--------------+
        |c1         |3            |a     |d    |1             |
        |c3         |1            |2     |4    |1             |
        +-----------+-------------+------+-----+--------------+
        <BLANKLINE>
        """
        print("Found the following differences:")
        diff_count_per_col_df.select(
            f.col("column_name"),
            f.col("total_nb_differences").alias("total_nb_diff"),
            f.col("left_value").alias(self.diff_format_options.left_df_alias),
            f.col("right_value").alias(self.diff_format_options.right_df_alias),
            f.col("nb_differences").alias("nb_differences"),
        ).show(1000 * self.diff_format_options.nb_diffed_rows, truncate=False)

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

    def _display_diff_examples(self, diff_df: DataFrame, diff_count_per_col_df: DataFrame, join_cols: List[str]):
        """For each column that has differences, print examples of rows where such a difference occurs.

        Examples:
            >>> from spark_frame.data_diff.diff_result_analyzer import _get_test_intersection_diff_df
            >>> _diff_df = _get_test_intersection_diff_df()
            >>> _diff_df.show()  # noqa: E501
            +---+-------------+-------------+
            | id|           c1|           c2|
            +---+-------------+-------------+
            |  1|{a, d, false}| {1, 1, true}|
            |  2|{b, a, false}|{2, 4, false}|
            +---+-------------+-------------+
            <BLANKLINE>
            >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
            >>> _diff_count_per_col_df = analyzer._get_diff_count_per_col_df(_diff_df, join_cols = ['id'])
            >>> analyzer._display_diff_examples(_diff_df, _diff_count_per_col_df, join_cols = ['id'])
            Detailed examples :
            'c1' : 2 rows
            +---+----------+---------+----------+---------+
            | id|before__c1|after__c1|before__c2|after__c2|
            +---+----------+---------+----------+---------+
            |  1|         a|        d|         1|        1|
            |  2|         b|        a|         2|        4|
            +---+----------+---------+----------+---------+
            <BLANKLINE>
            'c2' : 1 rows
            +---+----------+---------+----------+---------+
            | id|before__c1|after__c1|before__c2|after__c2|
            +---+----------+---------+----------+---------+
            |  2|         b|        a|         2|        4|
            +---+----------+---------+----------+---------+
            <BLANKLINE>
        """
        rows = (
            diff_count_per_col_df.select("column_name", "total_nb_differences")
            .distinct()
            .orderBy("column_name")
            .collect()
        )
        diff_count_per_col = [(r[0], r[1]) for r in rows]
        print("Detailed examples :")
        for col, nb in diff_count_per_col:
            print(f"'{col}' : {nb} rows")
            rows_that_changed_for_that_column = diff_df.filter(~diff_df[quote(col)]["is_equal"]).select(
                *join_cols, *[quote(r[0]) for r in rows]
            )
            self._format_diff_df(join_cols, rows_that_changed_for_that_column).show(
                self.diff_format_options.nb_diffed_rows
            )

    def _get_side_diff_df(self, diff_df: DataFrame, side: str, join_cols: List[str]) -> DataFrame:
        """Given a diff_df, compute the set of all values present only on the specified side.

        Examples:
            >>> from spark_frame.data_diff.package import _get_test_diff_df
            >>> _diff_df = _get_test_diff_df()
            >>> _diff_df.show()  # noqa: E501
            +---+----------------+----------------+-------------+------------+
            | id|              c1|              c2|   __EXISTS__|__IS_EQUAL__|
            +---+----------------+----------------+-------------+------------+
            |  1|    {a, a, true}|    {1, 1, true}| {true, true}|        true|
            |  2|    {b, b, true}|   {2, 4, false}| {true, true}|       false|
            |  3|{c, null, false}|{3, null, false}|{true, false}|       false|
            |  4|{null, f, false}|{null, 3, false}|{false, true}|       false|
            +---+----------------+----------------+-------------+------------+
            <BLANKLINE>
            >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
            >>> analyzer._get_side_diff_df(_diff_df, side="left", join_cols=["id"]).show()
            +---+---+---+
            | id| c1| c2|
            +---+---+---+
            |  3|  c|  3|
            +---+---+---+
            <BLANKLINE>
            >>> analyzer._get_side_diff_df(_diff_df, side="right", join_cols=["id"]).show()
            +---+---+---+
            | id| c1| c2|
            +---+---+---+
            |  4|  f|  3|
            +---+---+---+
            <BLANKLINE>
        """
        assert_true(side in ["left", "right"])
        if side == "left":
            predicate = PREDICATES.only_in_left
        else:
            predicate = PREDICATES.only_in_right
        df = diff_df.filter(predicate).drop(EXISTS_COL_NAME, IS_EQUAL_COL_NAME)
        compared_cols = [
            f.col(f"{field.name}.{side}_value").alias(field.name) for field in df.schema if field.name not in join_cols
        ]
        return df.select(*join_cols, *compared_cols)

    def _build_diff_count_per_col_df_from_unpivoted_diff_df(self, unpivoted_diff_df: DataFrame) -> DataFrame:
        """Given an `unpivoted_diff_df` DataFrame, builds a DataFrame that gives for each column the N most frequent
        differences that are happening, where N = `nb_diffed_rows`.

        Args:
            unpivoted_diff_df: A diff DataFrame
            nb_diffed_rows: A dict that gives for each column with differences the number of rows with changed values
              for this column

        Returns:

        Examples:
            >>> from pyspark.sql import SparkSession
            >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
            >>> unpivoted_diff_df = spark.sql( # noqa: E501
            ... '''
            ...     SELECT INLINE(ARRAY(
            ...         STRUCT(1 as id, "c1" as column_name, STRUCT("a" as left_value, "d" as right_value, False as is_equal) as diff),
            ...         STRUCT(1 as id, "c2" as column_name, STRUCT("x" as left_value, "x" as right_value, True as is_equal) as diff),
            ...         STRUCT(1 as id, "c3" as column_name, STRUCT("1" as left_value, "1" as right_value, True as is_equal) as diff),
            ...         STRUCT(2 as id, "c1" as column_name, STRUCT("b" as left_value, "a" as right_value, False as is_equal) as diff),
            ...         STRUCT(2 as id, "c2" as column_name, STRUCT("y" as left_value, "y" as right_value, True as is_equal) as diff),
            ...         STRUCT(2 as id, "c3" as column_name, STRUCT("2" as left_value, "4" as right_value, False as is_equal) as diff),
            ...         STRUCT(3 as id, "c1" as column_name, STRUCT("c" as left_value, "f" as right_value, False as is_equal) as diff),
            ...         STRUCT(3 as id, "c2" as column_name, STRUCT("z" as left_value, "z" as right_value, True as is_equal) as diff),
            ...         STRUCT(3 as id, "c3" as column_name, STRUCT("3" as left_value, "3" as right_value, True as is_equal) as diff)
            ...     ))
            ... ''')
            >>> unpivoted_diff_df.show()
            +---+-----------+-------------+
            | id|column_name|         diff|
            +---+-----------+-------------+
            |  1|         c1|{a, d, false}|
            |  1|         c2| {x, x, true}|
            |  1|         c3| {1, 1, true}|
            |  2|         c1|{b, a, false}|
            |  2|         c2| {y, y, true}|
            |  2|         c3|{2, 4, false}|
            |  3|         c1|{c, f, false}|
            |  3|         c2| {z, z, true}|
            |  3|         c3| {3, 3, true}|
            +---+-----------+-------------+
            <BLANKLINE>
            >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after", nb_diffed_rows=1))
            >>> analyzer._build_diff_count_per_col_df_from_unpivoted_diff_df(unpivoted_diff_df).orderBy('column_name').show()
            +-----------+----------+-----------+--------------+--------------------+
            |column_name|left_value|right_value|nb_differences|total_nb_differences|
            +-----------+----------+-----------+--------------+--------------------+
            |         c1|         a|          d|             1|                   3|
            |         c3|         2|          4|             1|                   1|
            +-----------+----------+-----------+--------------+--------------------+
            <BLANKLINE>
        """
        # We must make sure to break ties on nb_differences when ordering to ensure a deterministic for unit tests.
        window = Window.partitionBy(f.col("column_name")).orderBy(
            f.col("nb_differences").desc(), f.col("left_value"), f.col("right_value")
        )
        df = (
            unpivoted_diff_df.filter("COALESCE(diff.is_equal, FALSE) = FALSE")
            .groupBy("column_name", "diff.left_value", "diff.right_value")
            .agg(f.count(f.lit(1)).alias("nb_differences"))
            .withColumn("row_num", f.row_number().over(window))
            .withColumn("total_nb_differences", f.sum("nb_differences").over(Window.partitionBy("column_name")))
            .where(f.col("row_num") <= f.lit(self.diff_format_options.nb_diffed_rows))
            .drop("row_num")
        )
        return df.withColumn("column_name", f.regexp_replace(f.col("column_name"), STRUCT_SEPARATOR_REPLACEMENT, "."))

    def _get_top_per_col_state_df(
        self, diff_df: DataFrame, join_cols: List[str], max_nb_rows_per_col_state: Optional[int] = None
    ) -> DataFrame:
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
            max_nb_rows_per_col_state: Max number of rows to keep for each (column_name, state) tuple

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
            |  2|    {b, b, true}|   {2, 4, false}| {true, true}|       false|
            |  3|{c, null, false}|{3, null, false}|{true, false}|       false|
            |  4|{null, f, false}|{null, 3, false}|{false, true}|       false|
            +---+----------------+----------------+-------------+------------+
            <BLANKLINE>
            >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
            >>> (analyzer._get_top_per_col_state_df(_diff_df, join_cols = ['id'])
            ...  .orderBy("column_name", "state", "left_value")
            ... ).show()
            +-----------+-------------+----------+-----------+---+---------------+
            |column_name|        state|left_value|right_value| nb|col_state_total|
            +-----------+-------------+----------+-----------+---+---------------+
            |         c1|    no_change|         a|          a|  1|              2|
            |         c1|    no_change|         b|          b|  1|              2|
            |         c1| only_in_left|         c|       null|  1|              1|
            |         c1|only_in_right|      null|          f|  1|              1|
            |         c2|      changed|         2|          4|  1|              1|
            |         c2|    no_change|         1|          1|  1|              1|
            |         c2| only_in_left|         3|       null|  1|              1|
            |         c2|only_in_right|      null|          3|  1|              1|
            |         id|    no_change|         1|          1|  1|              2|
            |         id|    no_change|         2|          2|  1|              2|
            |         id| only_in_left|         3|       null|  1|              1|
            |         id|only_in_right|      null|          4|  1|              1|
            +-----------+-------------+----------+-----------+---+---------------+
            <BLANKLINE>

            *With `max_nb_rows_per_col_state=1`*
            >>> (analyzer._get_top_per_col_state_df(_diff_df, join_cols = ['id'], max_nb_rows_per_col_state=1)
            ...  .orderBy("column_name", "state", "left_value")
            ... ).show()
            +-----------+-------------+----------+-----------+---+---------------+
            |column_name|        state|left_value|right_value| nb|col_state_total|
            +-----------+-------------+----------+-----------+---+---------------+
            |         c1|    no_change|         a|          a|  1|              2|
            |         c1| only_in_left|         c|       null|  1|              1|
            |         c1|only_in_right|      null|          f|  1|              1|
            |         c2|      changed|         2|          4|  1|              1|
            |         c2|    no_change|         1|          1|  1|              1|
            |         c2| only_in_left|         3|       null|  1|              1|
            |         c2|only_in_right|      null|          3|  1|              1|
            |         id|    no_change|         1|          1|  1|              2|
            |         id| only_in_left|         3|       null|  1|              1|
            |         id|only_in_right|      null|          4|  1|              1|
            +-----------+-------------+----------+-----------+---+---------------+
            <BLANKLINE>
        """
        if max_nb_rows_per_col_state is None:
            _max_nb_rows_per_col_state = self.diff_format_options.nb_diffed_rows
        else:
            _max_nb_rows_per_col_state = max_nb_rows_per_col_state

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
        unpivoted_diff_df = self._unpivot(diff_df.drop(IS_EQUAL_COL_NAME), [EXISTS_COL_NAME])
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
            .withColumn("row_num", f.row_number().over(window))
            .withColumn("col_state_total", f.sum("nb").over(Window.partitionBy("column_name", "state")))
            .where(f.col("row_num") <= f.lit(_max_nb_rows_per_col_state))
            .drop("row_num")
        )
        return df

    def _get_diff_per_col_df_2(self, top_per_col_state_df: DataFrame, diff_result: DiffResult) -> DataFrame:
        """Given a diff_count_per_col_df, return a Dict[str, int] that gives for each column the total number
        of differences.

        Examples:
            >>> from spark_frame.data_diff.diff_results import _get_test_diff_result
            >>> diff_result = _get_test_diff_result()
            >>> diff_result.diff_df.show()
            +---+----------------+----------------+-------------+------------+
            | id|              c1|              c2|   __EXISTS__|__IS_EQUAL__|
            +---+----------------+----------------+-------------+------------+
            |  1|    {a, a, true}|    {1, 1, true}| {true, true}|        true|
            |  2|    {b, b, true}|   {2, 4, false}| {true, true}|       false|
            |  3|{c, null, false}|{3, null, false}|{true, false}|       false|
            |  4|{null, f, false}|{null, 3, false}|{false, true}|       false|
            +---+----------------+----------------+-------------+------------+
            <BLANKLINE>
            >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
            >>> top_per_col_state_df = analyzer._get_top_per_col_state_df(diff_result.diff_df, join_cols = ['id'])
            >>> top_per_col_state_df.show()
            +-----------+-------------+----------+-----------+---+---------------+
            |column_name|        state|left_value|right_value| nb|col_state_total|
            +-----------+-------------+----------+-----------+---+---------------+
            |         c1|    no_change|         a|          a|  1|              2|
            |         c1|    no_change|         b|          b|  1|              2|
            |         c1| only_in_left|         c|       null|  1|              1|
            |         c1|only_in_right|      null|          f|  1|              1|
            |         c2|      changed|         2|          4|  1|              1|
            |         c2|    no_change|         1|          1|  1|              1|
            |         c2| only_in_left|         3|       null|  1|              1|
            |         c2|only_in_right|      null|          3|  1|              1|
            |         id|    no_change|         1|          1|  1|              2|
            |         id|    no_change|         2|          2|  1|              2|
            |         id| only_in_left|         3|       null|  1|              1|
            |         id|only_in_right|      null|          4|  1|              1|
            +-----------+-------------+----------+-----------+---+---------------+
            <BLANKLINE>

            >>> df = analyzer._get_diff_per_col_df_2(top_per_col_state_df, diff_result)
            >>> from spark_frame import nested
            >>> nested.print_schema(df)
            root
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
            >>> df.show(truncate=False)
            +-----------+---------------+------------------------------------------------------------+
            |column_name|counts         |diff                                                        |
            +-----------+---------------+------------------------------------------------------------+
            |id         |{4, 0, 2, 1, 1}|{[], [{1, 1, 1}, {2, 2, 1}], [{3, null, 1}], [{null, 4, 1}]}|
            |c1         |{4, 0, 2, 1, 1}|{[], [{a, a, 1}, {b, b, 1}], [{c, null, 1}], [{null, f, 1}]}|
            |c2         |{4, 1, 1, 1, 1}|{[{2, 4, 1}], [{1, 1, 1}], [{3, null, 1}], [{null, 3, 1}]}  |
            +-----------+---------------+------------------------------------------------------------+
            <BLANKLINE>
        """
        diff_stats = diff_result.diff_stats
        columns = diff_result.diff_df.columns[:-2]
        col_df = diff_result.diff_df.sparkSession.createDataFrame(
            list(enumerate(columns)), "column_number INT, column_name STRING"
        )
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
            .agg(f.sum("nb").alias("nb"), f.expr("ARRAY_AGG(STRUCT(left_value, right_value, nb))").alias("diff"))
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
        )
        # df.show(truncate=False)
        # +-----------+------------------+------------------------------------------------------------+
        # |column_name|counts            |diff                                                        |
        # +-----------+------------------+------------------------------------------------------------+
        # |id         |{4, 0, 2, 0, 0, 0}|{[], [], [], []}                                            |
        # |c1         |{4, 0, 2, 1, 1, 1}|{[], [{a, a, 1}, {b, b, 1}], [{c, null, 1}], [{null, f, 1}]}|
        # |c2         |{4, 1, 1, 1, 1, 1}|{[{2, 4, 1}], [{1, 1, 1}], [{3, null, 1}], [{null, 3, 1}]}  |
        # +-----------+------------------+------------------------------------------------------------+
        return df

    def display_diff_results(self, diff_result: DiffResult, show_examples: bool):
        left_df_alias = self.diff_format_options.left_df_alias
        right_df_alias = self.diff_format_options.right_df_alias
        join_cols = diff_result.join_cols
        diff_stats = diff_result.diff_stats
        print_diff_stats(diff_result.diff_stats, left_df_alias, right_df_alias)
        if diff_stats.changed > 0:
            diff_count_per_col_df = self._get_diff_count_per_col_df(diff_result.changed_df, join_cols).persist()
            self._display_diff_count_per_col(diff_count_per_col_df)
            if show_examples:
                self._display_diff_examples(diff_result.diff_df, diff_count_per_col_df, join_cols)
        if diff_stats.only_in_left > 0:
            left_only_df = self._get_side_diff_df(diff_result.diff_df, "left", join_cols).persist()
            print(f"{diff_stats.only_in_left} rows were only found in '{left_df_alias}' :")
            analyze(left_only_df).show(100000)
        if diff_stats.only_in_right > 0:
            right_only_df = self._get_side_diff_df(diff_result.diff_df, "right", join_cols).persist()
            print(f"{diff_stats.only_in_right} rows were only found in '{right_df_alias}':")
            analyze(right_only_df).show(100000)

    def get_diff_result_summary(self, diff_result: DiffResult) -> DiffResultSummary:
        join_cols = diff_result.join_cols
        diff_count_per_col_df = self._get_top_per_col_state_df(diff_result.diff_df, join_cols).localCheckpoint()
        diff_per_col_df = self._get_diff_per_col_df_2(diff_count_per_col_df, diff_result)
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
