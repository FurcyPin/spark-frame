from typing import List, Optional

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from spark_frame.data_diff.diff_format_options import DiffFormatOptions
from spark_frame.data_diff.diff_per_col import _get_diff_per_col_df
from spark_frame.data_diff.diff_result import DiffResult
from spark_frame.data_diff.diff_result_summary import DiffResultSummary
from spark_frame.data_diff.diff_stats import print_diff_stats_shard
from spark_frame.data_diff.package import PREDICATES
from spark_frame.utils import MAX_JAVA_INT, quote


def _counts_changed_col() -> Column:
    """This method is used to make Sonar stop complaining about code duplicates"""
    return f.col("counts.changed")


def _diff_nb_col() -> Column:
    """This method is used to make Sonar stop complaining about code duplicates"""
    return f.col("diff.nb")


class DiffResultAnalyzer:
    def __init__(self, diff_format_options: Optional[DiffFormatOptions] = None) -> None:
        if diff_format_options is None:
            diff_format_options = DiffFormatOptions()
        self.diff_format_options = diff_format_options

    def _format_diff_df(self, join_cols: List[str], diff_df: DataFrame) -> DataFrame:
        """Given a diff DataFrame, rename the columns to prefix them with the left_df_alias and right_df_alias."""
        return diff_df.select(
            *[diff_df[quote(col_name)]["left_value"].alias(col_name) for col_name in join_cols],
            *[
                col
                for col_name in diff_df.columns
                if col_name not in join_cols
                for col in [
                    diff_df[quote(col_name)]["left_value"].alias(
                        f"{self.diff_format_options.left_df_alias}__{col_name}",
                    ),
                    diff_df[quote(col_name)]["right_value"].alias(
                        f"{self.diff_format_options.right_df_alias}__{col_name}",
                    ),
                ]
            ],
        )

    def _display_diff_examples(
        self,
        diff_df: DataFrame,
        diff_per_col_df: DataFrame,
        join_cols: List[str],
    ) -> None:
        """For each column that has differences, print examples of rows where such a difference occurs.

        Examples:
            >>> from spark_frame.data_diff.diff_result import _get_test_diff_result
            >>> diff_result = _get_test_diff_result()
            >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
            >>> diff_df = diff_result.diff_df_shards[""]
            >>> diff_df.show(truncate=False)
            +-----------------------------+-----------------------------+-----------------------------+---------------------------------+---------------------------------+-------------+------------+-------------+
            |id                           |c1                           |c2                           |c3                               |c4                               |__EXISTS__   |__IS_EQUAL__|__SAMPLE_ID__|
            +-----------------------------+-----------------------------+-----------------------------+---------------------------------+---------------------------------+-------------+------------+-------------+
            |{1, 1, true, true, true}     |{a, a, true, true, true}     |{1, 1, true, true, true}     |{1, NULL, false, true, false}    |{NULL, 1, false, false, true}    |{true, true} |true        |[{"id": 1}]  |
            |{2, 2, true, true, true}     |{b, b, true, true, true}     |{2, 3, false, true, true}    |{1, NULL, false, true, false}    |{NULL, 1, false, false, true}    |{true, true} |false       |[{"id": 2}]  |
            |{3, 3, true, true, true}     |{b, b, true, true, true}     |{2, 4, false, true, true}    |{2, NULL, false, true, false}    |{NULL, 2, false, false, true}    |{true, true} |false       |[{"id": 3}]  |
            |{4, 4, true, true, true}     |{b, b, true, true, true}     |{2, 4, false, true, true}    |{2, NULL, false, true, false}    |{NULL, 2, false, false, true}    |{true, true} |false       |[{"id": 4}]  |
            |{5, NULL, false, true, false}|{c, NULL, false, true, false}|{3, NULL, false, true, false}|{3, NULL, false, true, false}    |{NULL, NULL, false, false, false}|{true, false}|false       |[{"id": 5}]  |
            |{NULL, 6, false, false, true}|{NULL, f, false, false, true}|{NULL, 3, false, false, true}|{NULL, NULL, false, false, false}|{NULL, 3, false, false, true}    |{false, true}|false       |[{"id": 6}]  |
            +-----------------------------+-----------------------------+-----------------------------+---------------------------------+---------------------------------+-------------+------------+-------------+
            <BLANKLINE>
            >>> analyzer = DiffResultAnalyzer(DiffFormatOptions(left_df_alias="before", right_df_alias="after"))
            >>> diff_per_col_df = _get_test_diff_per_col_df()
            >>> diff_per_col_df.show(truncate=False)
            +-------------+-----------+---------------+----------------------------------------------------------------------------------------------------------------------------------------+
            |column_number|column_name|counts         |diff                                                                                                                                    |
            +-------------+-----------+---------------+----------------------------------------------------------------------------------------------------------------------------------------+
            |0            |id         |{6, 0, 4, 1, 1}|{[], [{1, 1, [{"id": 1}]}, {2, 1, [{"id": 2}]}, {3, 1, [{"id": 3}]}, {4, 1, [{"id": 4}]}], [{5, 1, [{"id": 5}]}], [{6, 1, [{"id": 6}]}]}|
            |1            |c1         |{6, 0, 4, 1, 1}|{[], [{a, 1, [{"id": 1}]}, {b, 3, [{"id": 2}]}], [{c, 1, [{"id": 5}]}], [{f, 1, [{"id": 6}]}]}                                          |
            |2            |c2         |{6, 3, 1, 1, 1}|{[{2, 3, 1, [{"id": 2}]}, {2, 4, 2, [{"id": 3}]}], [{1, 1, [{"id": 1}]}], [{3, 1, [{"id": 5}]}], [{3, 1, [{"id": 6}]}]}                 |
            |3            |c3         |{5, 0, 0, 5, 0}|{[], [], [{3, 1, [{"id": 5}]}, {1, 2, [{"id": 1}]}, {2, 2, [{"id": 3}]}], []}                                                           |
            |4            |c4         |{5, 0, 0, 0, 5}|{[], [], [], [{3, 1, [{"id": 6}]}, {1, 2, [{"id": 1}]}, {2, 2, [{"id": 3}]}]}                                                           |
            +-------------+-----------+---------------+----------------------------------------------------------------------------------------------------------------------------------------+
            <BLANKLINE>
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
        """  # noqa: E501
        rows = (
            diff_per_col_df.where(~f.col("column_name").isin(join_cols))
            .where(_counts_changed_col() > 0)
            .select("column_name", _counts_changed_col().alias("total_nb_differences"))
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
                self.diff_format_options.nb_top_values_kept_per_column,
            )

    @staticmethod
    def _display_changed(diff_per_col_df: DataFrame) -> None:
        """Displays the results of the diff analysis.

        We first display a summary of all columns that changed with the number of changes,
        then for each column, we display a summary of the most frequent changes and then
        we display examples of rows where this column changed, along with all the other columns
        that changed in this diff.

        Example:

        >>> diff_per_col_df = _get_test_diff_per_col_df()
        >>> diff_per_col_df.show(truncate=False)
        +-------------+-----------+---------------+----------------------------------------------------------------------------------------------------------------------------------------+
        |column_number|column_name|counts         |diff                                                                                                                                    |
        +-------------+-----------+---------------+----------------------------------------------------------------------------------------------------------------------------------------+
        |0            |id         |{6, 0, 4, 1, 1}|{[], [{1, 1, [{"id": 1}]}, {2, 1, [{"id": 2}]}, {3, 1, [{"id": 3}]}, {4, 1, [{"id": 4}]}], [{5, 1, [{"id": 5}]}], [{6, 1, [{"id": 6}]}]}|
        |1            |c1         |{6, 0, 4, 1, 1}|{[], [{a, 1, [{"id": 1}]}, {b, 3, [{"id": 2}]}], [{c, 1, [{"id": 5}]}], [{f, 1, [{"id": 6}]}]}                                          |
        |2            |c2         |{6, 3, 1, 1, 1}|{[{2, 3, 1, [{"id": 2}]}, {2, 4, 2, [{"id": 3}]}], [{1, 1, [{"id": 1}]}], [{3, 1, [{"id": 5}]}], [{3, 1, [{"id": 6}]}]}                 |
        |3            |c3         |{5, 0, 0, 5, 0}|{[], [], [{3, 1, [{"id": 5}]}, {1, 2, [{"id": 1}]}, {2, 2, [{"id": 3}]}], []}                                                           |
        |4            |c4         |{5, 0, 0, 0, 5}|{[], [], [], [{3, 1, [{"id": 6}]}, {1, 2, [{"id": 1}]}, {2, 2, [{"id": 3}]}]}                                                           |
        +-------------+-----------+---------------+----------------------------------------------------------------------------------------------------------------------------------------+
        <BLANKLINE>
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
         |-- diff.changed!.sample_ids!: string (nullable = true)
         |-- diff.no_change!.value: string (nullable = true)
         |-- diff.no_change!.nb: long (nullable = false)
         |-- diff.no_change!.sample_ids!: string (nullable = true)
         |-- diff.only_in_left!.value: string (nullable = true)
         |-- diff.only_in_left!.nb: long (nullable = false)
         |-- diff.only_in_left!.sample_ids!: string (nullable = true)
         |-- diff.only_in_right!.value: string (nullable = true)
         |-- diff.only_in_right!.nb: long (nullable = false)
         |-- diff.only_in_right!.sample_ids!: string (nullable = true)
        <BLANKLINE>
        >>> DiffResultAnalyzer._display_changed(diff_per_col_df)
        +-----------+-------------+----------+-----------+--------------+
        |column_name|total_nb_diff|left_value|right_value|nb_differences|
        +-----------+-------------+----------+-----------+--------------+
        |c2         |3            |2         |4          |2             |
        |c2         |3            |2         |3          |1             |
        +-----------+-------------+----------+-----------+--------------+
        <BLANKLINE>
        """  # noqa: E501
        df = diff_per_col_df.where(_counts_changed_col() > 0)
        df = df.select(
            "column_name",
            _counts_changed_col().alias("total_nb_diff"),
            f.explode("diff.changed").alias("diff"),
        ).orderBy("column_number", f.desc(_diff_nb_col()))
        df = df.select(
            "column_name",
            "total_nb_diff",
            "diff.left_value",
            "diff.right_value",
            _diff_nb_col().alias("nb_differences"),
        )
        df.show(MAX_JAVA_INT, truncate=False)

    @staticmethod
    def _display_only_in_left_or_right(
        diff_per_col_df: DataFrame,
        left_or_right: str,
    ) -> None:
        """Displays the results of the diff analysis.

        We first display a summary of all columns that changed with the number of changes,
        then for each column, we display a summary of the most frequent changes and then
        we display examples of rows where this column changed, along with all the other columns
        that changed in this diff.

        Example:

        >>> diff_per_col_df = _get_test_diff_per_col_df()
        >>> diff_per_col_df.show(truncate=False)
        +-------------+-----------+---------------+----------------------------------------------------------------------------------------------------------------------------------------+
        |column_number|column_name|counts         |diff                                                                                                                                    |
        +-------------+-----------+---------------+----------------------------------------------------------------------------------------------------------------------------------------+
        |0            |id         |{6, 0, 4, 1, 1}|{[], [{1, 1, [{"id": 1}]}, {2, 1, [{"id": 2}]}, {3, 1, [{"id": 3}]}, {4, 1, [{"id": 4}]}], [{5, 1, [{"id": 5}]}], [{6, 1, [{"id": 6}]}]}|
        |1            |c1         |{6, 0, 4, 1, 1}|{[], [{a, 1, [{"id": 1}]}, {b, 3, [{"id": 2}]}], [{c, 1, [{"id": 5}]}], [{f, 1, [{"id": 6}]}]}                                          |
        |2            |c2         |{6, 3, 1, 1, 1}|{[{2, 3, 1, [{"id": 2}]}, {2, 4, 2, [{"id": 3}]}], [{1, 1, [{"id": 1}]}], [{3, 1, [{"id": 5}]}], [{3, 1, [{"id": 6}]}]}                 |
        |3            |c3         |{5, 0, 0, 5, 0}|{[], [], [{3, 1, [{"id": 5}]}, {1, 2, [{"id": 1}]}, {2, 2, [{"id": 3}]}], []}                                                           |
        |4            |c4         |{5, 0, 0, 0, 5}|{[], [], [], [{3, 1, [{"id": 6}]}, {1, 2, [{"id": 1}]}, {2, 2, [{"id": 3}]}]}                                                           |
        +-------------+-----------+---------------+----------------------------------------------------------------------------------------------------------------------------------------+
        <BLANKLINE>
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
         |-- diff.changed!.sample_ids!: string (nullable = true)
         |-- diff.no_change!.value: string (nullable = true)
         |-- diff.no_change!.nb: long (nullable = false)
         |-- diff.no_change!.sample_ids!: string (nullable = true)
         |-- diff.only_in_left!.value: string (nullable = true)
         |-- diff.only_in_left!.nb: long (nullable = false)
         |-- diff.only_in_left!.sample_ids!: string (nullable = true)
         |-- diff.only_in_right!.value: string (nullable = true)
         |-- diff.only_in_right!.nb: long (nullable = false)
         |-- diff.only_in_right!.sample_ids!: string (nullable = true)
        <BLANKLINE>
        >>> DiffResultAnalyzer._display_only_in_left_or_right(diff_per_col_df, "left")
        +-----------+-----+---+
        |column_name|value|nb |
        +-----------+-----+---+
        |id         |5    |1  |
        |c1         |c    |1  |
        |c2         |3    |1  |
        |c3         |3    |1  |
        |c3         |1    |2  |
        |c3         |2    |2  |
        +-----------+-----+---+
        <BLANKLINE>
        >>> DiffResultAnalyzer._display_only_in_left_or_right(diff_per_col_df, "right")
        +-----------+-----+---+
        |column_name|value|nb |
        +-----------+-----+---+
        |id         |6    |1  |
        |c1         |f    |1  |
        |c2         |3    |1  |
        |c4         |3    |1  |
        |c4         |1    |2  |
        |c4         |2    |2  |
        +-----------+-----+---+
        <BLANKLINE>

        """  # noqa: E501
        df = diff_per_col_df.select(
            "column_name",
            f.explode(f"diff.only_in_{left_or_right}").alias("diff"),
        )
        df = df.select(
            "column_name",
            f.col("diff.value").alias("value"),
            _diff_nb_col().alias("nb"),
        )
        df.show(MAX_JAVA_INT, truncate=False)

    def display_diff_results(
        self,
        diff_result: DiffResult,
        show_examples: bool,
    ) -> None:
        join_cols = diff_result.join_cols
        diff_per_col_df = diff_result.get_diff_per_col_df(
            max_nb_rows_per_col_state=self.diff_format_options.nb_top_values_kept_per_column,
        )
        diff_stats_shards = diff_result.diff_stats_shards
        if diff_result.is_ok:
            print(f"\ndiff ok! ({diff_stats_shards[''].total} rows)\n")
            return
        print("\ndiff NOT ok\n")

        left_df_alias = self.diff_format_options.left_df_alias
        right_df_alias = self.diff_format_options.right_df_alias

        if len(diff_stats_shards) > 1:
            print(
                "WARNING: This diff has multiple granularity levels, "
                "we will print the results for each granularity level,\n"
                "         but we recommend to export the results to html for a much more digest result.\n",
            )

        for key, diff_stats_shard in diff_stats_shards.items():
            if len(diff_stats_shards) > 1:
                print("##############################################################")
                print(
                    f"Granularity : {'root' if key=='' else key} ({diff_stats_shard.total} rows)\n",
                )
            print_diff_stats_shard(diff_stats_shard, left_df_alias, right_df_alias)

            if diff_stats_shard.changed > 0:
                print("Found the following changes:")
                self._display_changed(diff_per_col_df)
                if show_examples:
                    self._display_diff_examples(
                        diff_result.diff_df_shards[key],
                        diff_per_col_df,
                        join_cols,
                    )
            if diff_stats_shard.only_in_left > 0:
                print(
                    f"{diff_stats_shard.only_in_left} rows were only found in '{left_df_alias}' :",
                )
                print(f"Most frequent values in '{left_df_alias}' for each column :")
                self._display_only_in_left_or_right(diff_per_col_df, "left")
            if diff_stats_shard.only_in_right > 0:
                print(
                    f"{diff_stats_shard.only_in_left} rows were only found in '{right_df_alias}' :",
                )
                print(f"Most frequent values in '{right_df_alias}' for each column :")
                self._display_only_in_left_or_right(diff_per_col_df, "right")

    def get_diff_result_summary(self, diff_result: DiffResult) -> DiffResultSummary:
        diff_per_col_df = diff_result.get_diff_per_col_df(
            max_nb_rows_per_col_state=self.diff_format_options.nb_top_values_kept_per_column,
        )
        sample_df_shards = diff_result.get_sample_df_shards(
            max_nb_rows_per_col_state=self.diff_format_options.nb_top_values_kept_per_column,
        )
        summary = DiffResultSummary(
            left_df_alias=self.diff_format_options.left_df_alias,
            right_df_alias=self.diff_format_options.right_df_alias,
            diff_per_col_df=diff_per_col_df,
            sample_df_shards=sample_df_shards,
            schema_diff_result=diff_result.schema_diff_result,
            join_cols=diff_result.join_cols,
            same_schema=diff_result.same_schema,
            same_data=diff_result.same_data,
            total_nb_rows=diff_result.total_nb_rows,
        )
        return summary

    def get_diff_per_col_df(
        self,
        diff_result: DiffResult,
    ) -> DataFrame:
        return diff_result.get_diff_per_col_df(self.diff_format_options.nb_top_values_kept_per_column)


def _get_test_diff_per_col_df() -> DataFrame:
    """Return an example of diff_per_col_df for testing purposes.
    We intentionally sort top_per_col_state_df by increasing "nb" to simulate the fact that we don't have
    any way to guarantee that the diff arrays will be sorted by decreasing order of "nb" in the `diff` column.
    """
    from spark_frame.data_diff.diff_result import _get_test_diff_result

    diff_result = _get_test_diff_result()
    df = _get_diff_per_col_df(
        top_per_col_state_df=diff_result.top_per_col_state_df.orderBy("nb"),
        columns=diff_result.schema_diff_result.column_names,
        max_nb_rows_per_col_state=10,
    )
    return df
