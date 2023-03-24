from typing import Dict, List, Optional, Tuple, TypeVar, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import IntegerType, LongType, StringType, StructField

from spark_frame.data_diff.diff_results import DiffResult
from spark_frame.data_diff.package import EXISTS_COL_NAME, IS_EQUAL_COL_NAME, STRUCT_SEPARATOR_REPLACEMENT, canonize_col
from spark_frame.data_diff.schema_diff import diff_dataframe_schemas
from spark_frame.data_type_utils import flatten_schema, get_common_columns, is_repeated
from spark_frame.transformations import flatten
from spark_frame.transformations_impl.convert_all_maps_to_arrays import convert_all_maps_to_arrays
from spark_frame.transformations_impl.harmonize_dataframes import harmonize_dataframes
from spark_frame.transformations_impl.sort_all_arrays import sort_all_arrays
from spark_frame.utils import quote, quote_columns


class DataframeComparatorException(Exception):
    pass


class CombinatorialExplosionError(DataframeComparatorException):
    pass


A = TypeVar("A")


class DataframeComparator:
    @staticmethod
    def _get_self_join_growth_estimate(df: DataFrame, cols: Union[str, List[str]]) -> float:
        """Computes how much time bigger a DataFrame will be if we self-join it using the provided columns, rounded
        to 2 decimals

        Args:
            df: A Spark DataFrame
            cols: A list of column names

        Returns:
            The estimated ratio of duplicates

        Examples:
            If a DataFrame with 6 rows has one value present on 2 rows and another value present on 3 rows,
            the growth factor will be (1*1 + 2*2 + 3*3) / 6 ~= 2.33.
            If a column unique on each row, it's number of duplicates will be 0.

            >>> from pyspark.sql import SparkSession
            >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
            >>> df = spark.sql('''SELECT INLINE(ARRAY(
            ...     STRUCT(1 as id, "a" as name),
            ...     STRUCT(2 as id, "b" as name),
            ...     STRUCT(3 as id, "b" as name),
            ...     STRUCT(4 as id, "c" as name),
            ...     STRUCT(5 as id, "c" as name),
            ...     STRUCT(6 as id, "c" as name)
            ... ))''')
            >>> DataframeComparator._get_self_join_growth_estimate(df, "id")
            1.0
            >>> DataframeComparator._get_self_join_growth_estimate(df, "name")
            2.33

        Tests:
            It should work with NULL values too.

            >>> df = spark.sql('''SELECT INLINE(ARRAY(
            ...     STRUCT(1 as id, "a" as name),
            ...     STRUCT(2 as id, "b" as name),
            ...     STRUCT(3 as id, "b" as name),
            ...     STRUCT(4 as id, NULL as name),
            ...     STRUCT(5 as id, NULL as name),
            ...     STRUCT(NULL as id, NULL as name)
            ... ))''')
            >>> DataframeComparator._get_self_join_growth_estimate(df, "id")
            1.0
            >>> DataframeComparator._get_self_join_growth_estimate(df, "name")
            2.33

        """
        if isinstance(cols, str):
            cols = [cols]
        df1 = df.groupby(quote_columns(cols)).agg(f.count(f.lit(1)).alias("nb"))
        df2 = df1.agg(
            f.sum(f.col("nb")).alias("nb_rows"), f.sum(f.col("nb") * f.col("nb")).alias("nb_rows_after_self_join")
        )
        res = df2.take(1)[0]
        nb_rows = res["nb_rows"]
        nb_rows_after_self_join = res["nb_rows_after_self_join"]
        if nb_rows_after_self_join is None:
            nb_rows_after_self_join = 0
        if nb_rows is None or nb_rows == 0:
            return 1.0
        else:
            return round(nb_rows_after_self_join * 1.0 / nb_rows, 2)

    @staticmethod
    def _get_eligible_columns_for_join(df: DataFrame) -> Dict[str, float]:
        """Identifies the column with the least duplicates, in order to use it as the id for the comparison join.

        Eligible columns are all columns of type String, Int or Bigint that have an approximate distinct count of 90%
        of the number of rows in the DataFrame. Returns None if no such column is found.

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''SELECT INLINE(ARRAY(
        ...     STRUCT(1 as id, "a" as name),
        ...     STRUCT(2 as id, "b" as name),
        ...     STRUCT(3 as id, "b" as name)
        ... ))''')
        >>> DataframeComparator._get_eligible_columns_for_join(df)
        {'id': 1.0}
        >>> df = spark.sql('''SELECT INLINE(ARRAY(
        ...     STRUCT(1 as id, "a" as name),
        ...     STRUCT(1 as id, "a" as name)
        ... ))''')
        >>> DataframeComparator._get_eligible_columns_for_join(df)
        {}

        :param df: a DataFrame
        :return: The name of the columns with less than 10% duplicates, and their
            corresponding self-join-growth-estimate
        """
        eligible_cols = [
            col.name
            for col in df.schema.fields
            if col.dataType in [StringType(), IntegerType(), LongType()] and not is_repeated(col)
        ]
        if len(eligible_cols) == 0:
            return dict()
        distinct_count_threshold = f.lit(90.0)
        eligibility_df = df.select(
            [
                (
                    f.when(f.count(f.lit(1)) == f.lit(0), f.lit(False)).otherwise(
                        f.approx_count_distinct(quote(col)) * f.lit(100.0) / f.count(f.lit(1))
                        > distinct_count_threshold
                    )
                ).alias(col)
                for col in eligible_cols
            ]
        )
        columns_with_high_distinct_count = [key for key, value in eligibility_df.collect()[0].asDict().items() if value]
        cols_with_duplicates = {
            col: DataframeComparator._get_self_join_growth_estimate(df, col) for col in columns_with_high_distinct_count
        }
        return cols_with_duplicates

    @staticmethod
    def _merge_growth_estimate_dicts(left_dict: Dict[str, float], right_dict: Dict[str, float]) -> Dict[str, float]:
        """Merge together two dicts giving for each column name the corresponding growth_estimate

        >>> DataframeComparator._merge_growth_estimate_dicts({"a": 10.0, "b": 1.0}, {"a": 1.0, "c": 1.0})
        {'a': 5.5, 'b': 1.0, 'c': 1.0}

        :param left_dict:
        :param right_dict:
        :return:
        """
        res = left_dict.copy()
        for x in right_dict:
            if x in left_dict:
                res[x] = (res[x] + right_dict[x]) / 2
            else:
                res[x] = right_dict[x]
        return res

    @staticmethod
    def _automatically_infer_join_col(left_df: DataFrame, right_df: DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """Identify the column with the least duplicates, in order to use it as the id for the comparison join.

        Eligible columns are all columns of type String, Int or Bigint that have an approximate distinct count of 90%
        of the number of rows in the DataFrame. Returns None if no suche column is found.

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> left_df = spark.sql('''SELECT INLINE(ARRAY(
        ...     STRUCT(1 as id, "a" as name),
        ...     STRUCT(2 as id, "b" as name),
        ...     STRUCT(3 as id, "c" as name),
        ...     STRUCT(4 as id, "d" as name),
        ...     STRUCT(5 as id, "e" as name),
        ...     STRUCT(6 as id, "f" as name)
        ... ))''')
        >>> right_df = spark.sql('''SELECT INLINE(ARRAY(
        ...     STRUCT(1 as id, "a" as name),
        ...     STRUCT(2 as id, "a" as name),
        ...     STRUCT(3 as id, "b" as name),
        ...     STRUCT(4 as id, "c" as name),
        ...     STRUCT(5 as id, "d" as name),
        ...     STRUCT(6 as id, "e" as name)
        ... ))''')
        >>> DataframeComparator._automatically_infer_join_col(left_df, right_df)
        ('id', 1.0)
        >>> left_df = spark.sql('''SELECT INLINE(ARRAY(
        ...     STRUCT(1 as id, "a" as name),
        ...     STRUCT(1 as id, "a" as name)
        ... ))''')
        >>> right_df = spark.sql('''SELECT INLINE(ARRAY(
        ...     STRUCT(1 as id, "a" as name),
        ...     STRUCT(1 as id, "a" as name)
        ... ))''')
        >>> DataframeComparator._automatically_infer_join_col(left_df, right_df)
        (None, None)

        :param left_df: a DataFrame
        :param right_df: a DataFrame
        :return: The name of the column with the least duplicates in both DataFrames if it has less than 10% duplicates.
        """
        left_col_dict = DataframeComparator._get_eligible_columns_for_join(left_df)
        right_col_dict = DataframeComparator._get_eligible_columns_for_join(right_df)
        merged_col_dict = DataframeComparator._merge_growth_estimate_dicts(left_col_dict, right_col_dict)

        if len(merged_col_dict) > 0:
            col, self_join_growth_estimate = sorted(merged_col_dict.items(), key=lambda x: x[1])[0]
            return col, self_join_growth_estimate
        else:
            return None, None

    def _get_join_cols(
        self, left_df: DataFrame, right_df: DataFrame, join_cols: Optional[List[str]]
    ) -> Tuple[List[str], float]:
        """Performs an in-depth analysis between two DataFrames with the same columns and prints the differences found.
        We first attempt to identify columns that look like ids.
        For that we choose all the columns with an approximate_count_distinct greater than 90% of the row count.
        For each column selected this way, we then perform a join and compare the DataFrames column by column.

        :param left_df: a DataFrame
        :param right_df: another DataFrame with the same columns
        :param join_cols: the list of columns on which to perform the join
        :return: a Dict that gives for each eligible join column the corresponding diff DataFrame
        """
        if join_cols is None:
            print(
                "No join_cols provided: "
                "trying to automatically infer a column that can be used for joining the two DataFrames"
            )
            inferred_join_col, self_join_growth_estimate = DataframeComparator._automatically_infer_join_col(
                left_df, right_df
            )
            if inferred_join_col is None or self_join_growth_estimate is None:
                raise DataframeComparatorException(
                    "Could not automatically infer a column sufficiently "
                    "unique to join the two DataFrames and perform a comparison. "
                    "Please specify manually the columns to use with the join_cols parameter"
                )
            else:
                print(f"Found the following column: {inferred_join_col}")
                join_cols = [inferred_join_col]
        else:
            self_join_growth_estimate = (
                DataframeComparator._get_self_join_growth_estimate(left_df, join_cols)
                + DataframeComparator._get_self_join_growth_estimate(right_df, join_cols)
            ) / 2
        return join_cols, self_join_growth_estimate

    def _check_join_cols(
        self, specified_join_cols: Optional[List[str]], join_cols: List[str], self_join_growth_estimate: float
    ) -> None:
        """Check the self_join_growth_estimate and raise an Exception if it is bigger than 2.

        This security helps to prevent users from accidentally spending huge query costs.
        Example: if a table has 10^9 rows and the join_col has a value with 10^6 duplicates, then the resulting
        self join will have (10^6)^2=10^12 which is 1000 times bigger than the original table.

        """
        inferred_provided_str = "provided"
        if specified_join_cols is None:
            inferred_provided_str = "inferred"
        if len(join_cols) == 1:
            plural_str = ""
            join_cols_str = str(join_cols[0])
        else:
            plural_str = "s"
            join_cols_str = str(join_cols)

        if self_join_growth_estimate >= 2.0:
            raise CombinatorialExplosionError(
                f"Performing a join with the {inferred_provided_str} column{plural_str} {join_cols_str} "
                f"would increase the size of the table by a factor of {self_join_growth_estimate}. "
                f"Please provide join_cols that are truly unique for both DataFrames."
            )
        print(
            f"We will try to find the differences by joining the DataFrames together "
            f"using the {inferred_provided_str} column{plural_str}: {join_cols_str}"
        )
        if self_join_growth_estimate > 1.0:
            print(
                f"WARNING: duplicates have been detected in the joining key, the resulting DataFrame "
                f"will be {self_join_growth_estimate} bigger which might affect the diff results. "
                f"Please consider providing join_cols that are truly unique for both DataFrames."
            )

    @staticmethod
    def _build_null_safe_join_clause(left_df: DataFrame, right_df: DataFrame, join_cols: List[str]) -> Column:
        """Generates a join clause that matches NULL values for the given join_cols"""

        def join_clause_for_single_column(column: str) -> Column:
            return left_df[column].eqNullSafe(right_df[column])

        first_column: str = join_cols[0]
        join_clause: Column = join_clause_for_single_column(first_column)
        for col in join_cols[1:]:
            join_clause &= join_clause_for_single_column(col)
        return join_clause

    @staticmethod
    def _build_diff_dataframe(left_df: DataFrame, right_df: DataFrame, join_cols: List[str]) -> DataFrame:
        """Perform a column-by-column comparison between two DataFrames.
        The two DataFrames must have the same columns with the same ordering.
        The column `join_col` will be used to join the two DataFrames together.
        Then we build a new DataFrame with the `join_col` and for each column, a struct with three elements:
        - `left_value`: the value coming from the `left_df`
        - `right_value`: the value coming from the `right_df`
        - `is_equal`: True if both values have the same hash, False otherwise.

        Example:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> left_df = spark.sql('''SELECT INLINE(ARRAY(
        ...     STRUCT(1 as id, "a" as c1, 1 as c2),
        ...     STRUCT(2 as id, "b" as c1, 2 as c2),
        ...     STRUCT(3 as id, "c" as c1, 3 as c2)
        ... ))''')
        >>> right_df = spark.sql('''SELECT INLINE(ARRAY(
        ...     STRUCT(1 as id, "a" as c1, 1 as c2),
        ...     STRUCT(2 as id, "b" as c1, 4 as c2),
        ...     STRUCT(4 as id, "f" as c1, 3 as c2)
        ... ))''')
        >>> left_df.show()
        +---+---+---+
        | id| c1| c2|
        +---+---+---+
        |  1|  a|  1|
        |  2|  b|  2|
        |  3|  c|  3|
        +---+---+---+
        <BLANKLINE>
        >>> right_df.show()
        +---+---+---+
        | id| c1| c2|
        +---+---+---+
        |  1|  a|  1|
        |  2|  b|  4|
        |  4|  f|  3|
        +---+---+---+
        <BLANKLINE>
        >>> DataframeComparator._build_diff_dataframe(left_df, right_df, ['id']).orderBy('id').show()  # noqa: E501
        +---+----------------+----------------+-------------+------------+
        | id|              c1|              c2|   __EXISTS__|__IS_EQUAL__|
        +---+----------------+----------------+-------------+------------+
        |  1|    {a, a, true}|    {1, 1, true}| {true, true}|        true|
        |  2|    {b, b, true}|   {2, 4, false}| {true, true}|       false|
        |  3|{c, null, false}|{3, null, false}|{true, false}|       false|
        |  4|{null, f, false}|{null, 3, false}|{false, true}|       false|
        +---+----------------+----------------+-------------+------------+
        <BLANKLINE>


        :param left_df: a DataFrame
        :param right_df: a DataFrame with the same columns
        :param join_cols: the columns to use to perform the join.
        :return: a DataFrame containing all the columns that differ, and a dictionary that gives the number of
            differing rows for each column
        """
        left_df = left_df.withColumn(EXISTS_COL_NAME, f.lit(True))
        right_df = right_df.withColumn(EXISTS_COL_NAME, f.lit(True))

        null_safe_join_clause = DataframeComparator._build_null_safe_join_clause(left_df, right_df, join_cols)
        diff = left_df.join(right_df, null_safe_join_clause, "full")

        compared_fields = [
            field for field in left_df.schema if field.name not in join_cols and field.name != EXISTS_COL_NAME
        ]

        def comparison_struct(field: StructField) -> Column:
            left_col: Column = left_df[field.name]
            right_col: Column = right_df[field.name]
            left_col_str: Column = canonize_col(left_col, field)
            right_col_str: Column = canonize_col(right_col, field)
            return f.struct(
                left_col.alias("left_value"),
                right_col.alias("right_value"),
                (
                    (left_col_str.isNull() & right_col_str.isNull())
                    | (left_col_str.isNotNull() & right_col_str.isNotNull() & (left_col_str == right_col_str))
                ).alias("is_equal"),
            ).alias(field.name)

        diff_df = diff.select(
            *[f.coalesce(left_df[col], right_df[col]).alias(col) for col in join_cols],
            *[comparison_struct(field) for field in compared_fields],
            f.struct(
                f.coalesce(left_df[EXISTS_COL_NAME], f.lit(False)).alias("left_value"),
                f.coalesce(right_df[EXISTS_COL_NAME], f.lit(False)).alias("right_value"),
            ).alias(EXISTS_COL_NAME),
        )

        row_is_equal = f.lit(True)
        for field in compared_fields:
            row_is_equal = row_is_equal & f.col(f"{field.name}.is_equal")
        return diff_df.withColumn(IS_EQUAL_COL_NAME, row_is_equal)

    def _harmonize_and_normalize_dataframes(
        self,
        left_flat: DataFrame,
        right_flat: DataFrame,
        common_column_shard: List[Tuple[str, Optional[str]]],
        skip_make_dataframes_comparable: bool,
    ) -> Tuple[DataFrame, DataFrame]:
        if not skip_make_dataframes_comparable:
            left_flat, right_flat = harmonize_dataframes(left_flat, right_flat, common_column_shard)
        left_flat = sort_all_arrays(left_flat)
        right_flat = sort_all_arrays(right_flat)
        return left_flat, right_flat

    def compare_df(self, left_df: DataFrame, right_df: DataFrame, join_cols: Optional[List[str]] = None) -> DiffResult:
        """Compares two DataFrames and print out the differences.

        We first compare the DataFrame schemas. If the schemas are different, we adapt the DataFrames to make them
        as much comparable as possible:
        - If the column ordering changed, we re-order them
        - If a column type changed, we cast the column to the smallest common type
        - If a column was added, removed or renamed, it will be ignored.

        If `join_cols` is specified, we will use the specified columns to perform the comparison join between the
        two DataFrames. Ideally, the `join_cols` should respect an unicity constraint.
        If they contain duplicates, a safety check is performed to prevent a potential combinatorial explosion:
        if the number of rows in the joined DataFrame would be more than twice the size of the original DataFrames,
        then an Exception is raised and the user will be asked to provide another set of `join_cols`.

        If no `join_cols` is specified, the algorithm will try to automatically find a single column suitable for
        the join. However, the automatic inference can only find join keys based on a single column.
        If the DataFrame's unique keys are composite (multiple columns) they must be given explicitly via `join_cols`
        to perform the diff analysis.

        !!! Tips:

            - If you want to test a column renaming, you can temporarily add renaming step to the DataFrame
              you want to test.
            - When comparing arrays, this algorithm ignores their ordering (e.g. `[1, 2, 3] == [3, 2, 1]`)
            - The algorithm is able to handle nested non-repeated records, such as STRUCT<STRUCT<>>
              or even ARRAY<STRUCT>>, but it doesn't support nested repeated structures,
              such as ARRAY<STRUCT<ARRAY<>>>.

        Args:
            left_df: A Spark DataFrame
            right_df: Another DataFrame
            join_cols: Specifies the columns on which the two DataFrames should be joined to compare them

        Returns:
            A DiffResult object

        Examples:
            >>> from spark_frame.data_diff.dataframe_comparator import __get_test_dfs
            >>> df1, df2 = __get_test_dfs()
            >>> diff_result = DataframeComparator().compare_df(df1, df2)  # noqa: E501
            <BLANKLINE>
            Analyzing differences...
            No join_cols provided: trying to automatically infer a column that can be used for joining the two DataFrames
            Found the following column: id
            We will try to find the differences by joining the DataFrames together using the inferred column: id
            >>> diff_result.display()
            Schema has changed:
            @@ -1,4 +1,5 @@
            <BLANKLINE>
             id INT
             my_array!.a INT
             my_array!.b INT
             my_array!.c INT
            +my_array!.d INT
            WARNING: columns that do not match both sides will be ignored
            <BLANKLINE>
            diff NOT ok
            <BLANKLINE>
            Summary:
            <BLANKLINE>
            Row count ok: 3 rows
            <BLANKLINE>
            1 (25.0%) rows are identical
            1 (25.0%) rows have changed
            1 (25.0%) rows are only in 'left'
            1 (25.0%) rows are only in 'right
            <BLANKLINE>
            Found the following changes:
            +-----------+-------------+---------------------+---------------------+--------------+
            |column_name|total_nb_diff|left_value           |right_value          |nb_differences|
            +-----------+-------------+---------------------+---------------------+--------------+
            |my_array   |1            |[{"a":1,"b":2,"c":3}]|[{"a":2,"b":2,"c":3}]|1             |
            +-----------+-------------+---------------------+---------------------+--------------+
            <BLANKLINE>
            1 rows were only found in 'left' :
            Most frequent values in 'left' for each column :
            +-----------+---------------------+---+
            |column_name|value                |nb |
            +-----------+---------------------+---+
            |id         |3                    |1  |
            |my_array   |[{"a":1,"b":2,"c":3}]|1  |
            +-----------+---------------------+---+
            <BLANKLINE>
            1 rows were only found in 'right' :
            Most frequent values in 'right' for each column :
            +-----------+---------------------+---+
            |column_name|value                |nb |
            +-----------+---------------------+---+
            |id         |4                    |1  |
            |my_array   |[{"a":1,"b":2,"c":3}]|1  |
            +-----------+---------------------+---+
            <BLANKLINE>
        """
        if join_cols == []:
            join_cols = None
        specified_join_cols = join_cols
        left_df = convert_all_maps_to_arrays(left_df)
        right_df = convert_all_maps_to_arrays(right_df)

        schema_diff_result = diff_dataframe_schemas(left_df, right_df)

        left_flat = flatten(left_df, struct_separator=STRUCT_SEPARATOR_REPLACEMENT)
        right_flat = flatten(right_df, struct_separator=STRUCT_SEPARATOR_REPLACEMENT)

        print("\nAnalyzing differences...")
        join_cols, self_join_growth_estimate = self._get_join_cols(left_flat, right_flat, join_cols)
        self._check_join_cols(specified_join_cols, join_cols, self_join_growth_estimate)

        left_schema_flat = flatten_schema(left_flat.schema, explode=True)
        if not schema_diff_result.same_schema:
            right_schema_flat = flatten_schema(right_flat.schema, explode=True)
            common_columns = get_common_columns(left_schema_flat, right_schema_flat)
        else:
            common_columns = [(field.name, None) for field in left_schema_flat]

        left_flat, right_flat = self._harmonize_and_normalize_dataframes(
            left_flat, right_flat, common_columns, skip_make_dataframes_comparable=schema_diff_result.same_schema
        )
        diff_df = self._build_diff_dataframe(left_flat, right_flat, join_cols)

        return DiffResult(schema_diff_result, diff_df, join_cols)


def __get_test_dfs() -> Tuple[DataFrame, DataFrame]:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("doctest").getOrCreate()

    df1 = spark.sql(
        """
        SELECT INLINE (ARRAY(
            STRUCT(1 as id, ARRAY(STRUCT(1 as a, 2 as b, 3 as c)) as my_array),
            STRUCT(2 as id, ARRAY(STRUCT(1 as a, 2 as b, 3 as c)) as my_array),
            STRUCT(3 as id, ARRAY(STRUCT(1 as a, 2 as b, 3 as c)) as my_array)
        ))
    """
    )
    df2 = spark.sql(
        """
        SELECT INLINE (ARRAY(
            STRUCT(1 as id, ARRAY(STRUCT(1 as a, 2 as b, 3 as c, 4 as d)) as my_array),
            STRUCT(2 as id, ARRAY(STRUCT(2 as a, 2 as b, 3 as c, 4 as d)) as my_array),
            STRUCT(4 as id, ARRAY(STRUCT(1 as a, 2 as b, 3 as c, 4 as d)) as my_array)
       ))
    """
    )
    return df1, df2
