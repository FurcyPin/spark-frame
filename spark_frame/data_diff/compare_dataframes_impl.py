from typing import Dict, List, Optional, Tuple, TypeVar, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import IntegerType, LongType, StringType

from spark_frame import nested
from spark_frame.data_diff.diff_result import DiffResult
from spark_frame.data_diff.package import (
    EXISTS_COL_NAME,
    IS_EQUAL_COL_NAME,
    SAMPLE_ID_COL_NAME,
    STRUCT_SEPARATOR_REPLACEMENT,
    canonize_col,
)
from spark_frame.data_diff.schema_diff import (
    DiffPrefix,
    SchemaDiffResult,
    diff_dataframe_schemas,
)
from spark_frame.data_diff.special_characters import (
    _replace_special_characters,
    _replace_special_characters_from_col_names,
    _restore_special_characters,
)
from spark_frame.data_type_utils import is_repeated
from spark_frame.exceptions import CombinatorialExplosionError, DataframeComparisonException
from spark_frame.field_utils import has_same_or_higher_granularity
from spark_frame.nested_impl.package import unnest_fields, validate_fields_exist
from spark_frame.transformations import flatten
from spark_frame.transformations_impl.convert_all_maps_to_arrays import (
    convert_all_maps_to_arrays,
)
from spark_frame.transformations_impl.harmonize_dataframes import harmonize_dataframes
from spark_frame.transformations_impl.sort_all_arrays import sort_all_arrays
from spark_frame.utils import quote, quote_columns

A = TypeVar("A")


def _deduplicate_list_while_conserving_ordering(a_list: List[A]) -> List[A]:
    """Deduplicate a list while conserving its ordering.
    The implementation uses the fact that unlike sets, dict keys preserve ordering.

    Args:
        a_list: A list

    Returns: A deduplicated list

    Examples:
        >>> _deduplicate_list_while_conserving_ordering([1, 3, 2, 1, 2, 3, 1, 4])
        [1, 3, 2, 4]

    """
    return list({elem: 0 for elem in a_list}.keys())


def _get_common_root_column_names(common_fields: Dict[str, Optional[str]]) -> List[str]:
    """Given common_columns, compute the ordered list of names of common root columns.

    Args:
        common_fields: the list of common fields

    Returns:

    Examples:
        >>> _get_common_root_column_names({"id": None, "array!.c1": None, "array!.c2": None})
        ['id', 'array']

    """
    root_columns = [col.split("!")[0] for col in common_fields]
    return _deduplicate_list_while_conserving_ordering(root_columns)


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
        >>> _get_self_join_growth_estimate(df, "id")
        1.0
        >>> _get_self_join_growth_estimate(df, "name")
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
        >>> _get_self_join_growth_estimate(df, "id")
        1.0
        >>> _get_self_join_growth_estimate(df, "name")
        2.33

    """
    if isinstance(cols, str):
        cols = [cols]
    df1 = df.groupBy(*quote_columns(cols)).agg(f.count(f.lit(1)).alias("nb"))
    df2 = df1.agg(
        f.sum(f.col("nb")).alias("nb_rows"),
        f.sum(f.col("nb") * f.col("nb")).alias("nb_rows_after_self_join"),
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


def _get_eligible_columns_for_join(df: DataFrame) -> Dict[str, float]:
    """Identifies the column with the least duplicates, in order to use it as the id for the comparison join.

    Eligible columns are all columns of type String, Int or Bigint that have an approximate distinct count of 90%
    of the number of rows in the DataFrame. Returns None if no such column is found.

    Args:
        df: a DataFrame

    Returns:
        The name of the columns with less than 10% duplicates, and their
        corresponding self-join-growth-estimate

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''SELECT INLINE(ARRAY(
        ...     STRUCT(1 as id, "a" as name),
        ...     STRUCT(2 as id, "b" as name),
        ...     STRUCT(3 as id, "b" as name)
        ... ))''')
        >>> _get_eligible_columns_for_join(df)
        {'id': 1.0}
        >>> df = spark.sql('''SELECT INLINE(ARRAY(
        ...     STRUCT(1 as id, "a" as name),
        ...     STRUCT(1 as id, "a" as name)
        ... ))''')
        >>> _get_eligible_columns_for_join(df)
        {}
    """
    eligible_cols = [
        col.name
        for col in df.schema.fields
        if col.dataType in [StringType(), IntegerType(), LongType()] and not is_repeated(col)
    ]
    if len(eligible_cols) == 0:
        return {}
    distinct_count_threshold = f.lit(90.0)
    eligibility_df = df.select(
        [
            (
                f.when(f.count(f.lit(1)) == f.lit(0), f.lit(col=False)).otherwise(
                    f.approx_count_distinct(quote(col)) * f.lit(100.0) / f.count(f.lit(1)) > distinct_count_threshold,
                )
            ).alias(col)
            for col in eligible_cols
        ],
    )
    columns_with_high_distinct_count = [key for key, value in eligibility_df.collect()[0].asDict().items() if value]
    cols_with_duplicates = {col: _get_self_join_growth_estimate(df, col) for col in columns_with_high_distinct_count}
    return cols_with_duplicates


def _merge_growth_estimate_dicts(
    left_dict: Dict[str, float],
    right_dict: Dict[str, float],
) -> Dict[str, float]:
    """Merge together two dicts giving for each column name the corresponding growth_estimate

    >>> _merge_growth_estimate_dicts({"a": 10.0, "b": 1.0}, {"a": 1.0, "c": 1.0})
    {'a': 5.5, 'b': 1.0, 'c': 1.0}
    """
    res = left_dict.copy()
    for x in right_dict:
        if x in left_dict:
            res[x] = (res[x] + right_dict[x]) / 2
        else:
            res[x] = right_dict[x]
    return res


def _automatically_infer_join_col(
    left_df: DataFrame,
    right_df: DataFrame,
) -> Tuple[Optional[str], Optional[float]]:
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
    >>> _automatically_infer_join_col(left_df, right_df)
    ('id', 1.0)
    >>> left_df = spark.sql('''SELECT INLINE(ARRAY(
    ...     STRUCT(1 as id, "a" as name),
    ...     STRUCT(1 as id, "a" as name)
    ... ))''')
    >>> right_df = spark.sql('''SELECT INLINE(ARRAY(
    ...     STRUCT(1 as id, "a" as name),
    ...     STRUCT(1 as id, "a" as name)
    ... ))''')
    >>> _automatically_infer_join_col(left_df, right_df)
    (None, None)

    :param left_df: a DataFrame
    :param right_df: a DataFrame
    :return: The name of the column with the least duplicates in both DataFrames if it has less than 10% duplicates.
    """
    left_col_dict = _get_eligible_columns_for_join(left_df)
    right_col_dict = _get_eligible_columns_for_join(right_df)
    merged_col_dict = _merge_growth_estimate_dicts(left_col_dict, right_col_dict)

    if len(merged_col_dict) > 0:
        col, self_join_growth_estimate = sorted(
            merged_col_dict.items(),
            key=lambda x: x[1],
        )[0]
        return col, self_join_growth_estimate
    else:
        return None, None


def _get_join_cols(
    left_df: DataFrame,
    right_df: DataFrame,
    join_cols: Optional[List[str]],
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
            "trying to automatically infer a column that can be used for joining the two DataFrames",
        )
        inferred_join_col, self_join_growth_estimate = _automatically_infer_join_col(
            left_df,
            right_df,
        )
        if inferred_join_col is None or self_join_growth_estimate is None:
            error_message = (
                "Could not automatically infer a column sufficiently "
                "unique to join the two DataFrames and perform a comparison. "
                "Please specify manually the columns to use with the join_cols parameter"
            )
            raise DataframeComparisonException(error_message)
        else:
            print(f"Found the following column: {inferred_join_col}")
            join_cols = [inferred_join_col]
    else:
        self_join_growth_estimate = (
            _get_self_join_growth_estimate(left_df, join_cols) + _get_self_join_growth_estimate(right_df, join_cols)
        ) / 2
    return join_cols, self_join_growth_estimate


def _check_join_cols(
    specified_join_cols: Optional[List[str]],
    join_cols: List[str],
    self_join_growth_estimate: float,
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
    join_cols_str = _restore_special_characters(join_cols_str)

    if self_join_growth_estimate >= 2.0:
        error_message = (
            f"Performing a join with the {inferred_provided_str} column{plural_str} {join_cols_str} "
            f"would increase the size of the table by a factor of {self_join_growth_estimate}. "
            f"Please provide join_cols that are truly unique for both DataFrames."
        )
        raise CombinatorialExplosionError(error_message)
    print(
        f"Generating the diff by joining the DataFrames together "
        f"using the {inferred_provided_str} column{plural_str}: {join_cols_str}",
    )
    if self_join_growth_estimate > 1.0:
        print(
            f"WARNING: duplicates have been detected in the joining key, the resulting DataFrame "
            f"will be {self_join_growth_estimate} bigger which might affect the diff results. "
            f"Please consider providing join_cols that are truly unique for both DataFrames.",
        )


def _build_null_safe_join_clause(
    left_df: DataFrame,
    right_df: DataFrame,
    join_cols: List[str],
) -> Column:
    """Generates a join clause that matches NULL values for the given join_cols"""

    def join_clause_for_single_column(column: str) -> Column:
        return left_df[column].eqNullSafe(right_df[column])

    first_column: str = join_cols[0]
    join_clause: Column = join_clause_for_single_column(first_column)
    for col in join_cols[1:]:
        join_clause &= join_clause_for_single_column(col)
    return join_clause


def _build_diff_dataframe(
    left_df: DataFrame,
    right_df: DataFrame,
    column_names_diff: Dict[str, DiffPrefix],
    join_cols: List[str],
    granularities: List[str],
) -> DataFrame:
    """Perform a column-by-column comparison between two DataFrames.
    The two DataFrames must have the same columns with the same ordering.
    The column `join_col` will be used to join the two DataFrames together.
    Then we build a new DataFrame with the `join_col` and for each column, a struct with three elements:
    - `left_value`: the value coming from the `left_df`
    - `right_value`: the value coming from the `right_df`
    - `is_equal`: True if both values have the same hash, False otherwise.

    Args:
        left_df: A DataFrame
        right_df: Another DataFrame
        join_cols: The names of the columns to use to perform the join.

    Returns:
        A DataFrame containing all the columns that differ, and a dictionary that gives the number of
        differing rows for each column

    Examples:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> left_df = spark.sql('''SELECT INLINE(ARRAY(
        ...     STRUCT(1 as id, "a" as c1, 1 as c2, 2 as c3),
        ...     STRUCT(2 as id, "b" as c1, 2 as c2, 3 as c3),
        ...     STRUCT(3 as id, "c" as c1, 3 as c2, 4 as c3)
        ... ))''')
        >>> right_df = spark.sql('''SELECT INLINE(ARRAY(
        ...     STRUCT(1 as id, "a" as c1, 1 as c2, 3 as c4),
        ...     STRUCT(2 as id, "b" as c1, 4 as c2, 4 as c4),
        ...     STRUCT(4 as id, "f" as c1, 3 as c2, 5 as c4)
        ... ))''')
        >>> left_df.show()
        +---+---+---+---+
        | id| c1| c2| c3|
        +---+---+---+---+
        |  1|  a|  1|  2|
        |  2|  b|  2|  3|
        |  3|  c|  3|  4|
        +---+---+---+---+
        <BLANKLINE>
        >>> right_df.show()
        +---+---+---+---+
        | id| c1| c2| c4|
        +---+---+---+---+
        |  1|  a|  1|  3|
        |  2|  b|  4|  4|
        |  4|  f|  3|  5|
        +---+---+---+---+
        <BLANKLINE>
        >>> from spark_frame.data_diff.schema_diff import _diff_dataframe_column_names
        >>> column_names_diff = _diff_dataframe_column_names(left_df.columns, right_df.columns)
        >>> column_names_diff
        {'id': ' ', 'c1': ' ', 'c2': ' ', 'c3': '-', 'c4': '+'}
        >>> (_build_diff_dataframe(left_df, right_df, column_names_diff, join_cols=['id'], granularities=[""])
        ...     .withColumn('coalesced_id', f.expr('coalesce(id.left_value, id.right_value)'))
        ...     .orderBy('coalesced_id').drop('coalesced_id').show(truncate=False)
        ... ) # noqa: E501
        +-----------------------------+-----------------------------+-----------------------------+---------------------------------+---------------------------------+-------------+------------+-------------+
        |id                           |c1                           |c2                           |c3                               |c4                               |__EXISTS__   |__IS_EQUAL__|__SAMPLE_ID__|
        +-----------------------------+-----------------------------+-----------------------------+---------------------------------+---------------------------------+-------------+------------+-------------+
        |{1, 1, true, true, true}     |{a, a, true, true, true}     |{1, 1, true, true, true}     |{2, NULL, false, true, false}    |{NULL, 3, false, false, true}    |{true, true} |true        |[{"id":1}]   |
        |{2, 2, true, true, true}     |{b, b, true, true, true}     |{2, 4, false, true, true}    |{3, NULL, false, true, false}    |{NULL, 4, false, false, true}    |{true, true} |false       |[{"id":2}]   |
        |{3, NULL, false, true, false}|{c, NULL, false, true, false}|{3, NULL, false, true, false}|{4, NULL, false, true, false}    |{NULL, NULL, false, false, false}|{true, false}|false       |[{"id":3}]   |
        |{NULL, 4, false, false, true}|{NULL, f, false, false, true}|{NULL, 3, false, false, true}|{NULL, NULL, false, false, false}|{NULL, 5, false, false, true}    |{false, true}|false       |[{"id":4}]   |
        +-----------------------------+-----------------------------+-----------------------------+---------------------------------+---------------------------------+-------------+------------+-------------+
        <BLANKLINE>
    """  # noqa: E501
    column_names_diff = {_replace_special_characters(col_name): diff for col_name, diff in column_names_diff.items()}

    left_df = left_df.withColumn(EXISTS_COL_NAME, f.lit(col=True))
    right_df = right_df.withColumn(EXISTS_COL_NAME, f.lit(col=True))

    null_safe_join_clause = _build_null_safe_join_clause(left_df, right_df, join_cols)
    diff = left_df.join(right_df, null_safe_join_clause, "full")

    left_df_fields = {field.name: field for field in left_df.schema.fields}
    right_df_fields = {field.name: field for field in right_df.schema.fields}

    def comparison_struct(col_name: str, diff_prefix: str) -> Column:
        if diff_prefix == DiffPrefix.ADDED:
            # We use IF(true, None, right_df.col_name) to force
            # the type of the NULL to be the same as right_df.col_name.
            left_col = f.when(f.lit(col=True), f.lit(col=None)).otherwise(right_df[col_name])
            left_col_str = left_col
            exists_left = f.lit(col=False)
        else:
            left_col = left_df[col_name]
            left_col_str = canonize_col(left_col, left_df_fields[col_name])
            exists_left = f.coalesce(left_df[EXISTS_COL_NAME], f.lit(col=False))

        if diff_prefix == DiffPrefix.REMOVED:
            # We use IF(true, None, left_df.col_name) to force
            # the type of the NULL to be the same as left_df.col_name.
            right_col = f.when(f.lit(col=True), f.lit(col=None)).otherwise(left_df[col_name])
            right_col_str = right_col
            exists_right = f.lit(col=False)
        else:
            right_col = right_df[col_name]
            right_col_str = canonize_col(right_col, right_df_fields[col_name])
            exists_right = f.coalesce(right_df[EXISTS_COL_NAME], f.lit(col=False))

        if diff_prefix == DiffPrefix.UNCHANGED:
            is_equal_col = (left_col_str.isNull() & right_col_str.isNull()) | (
                left_col_str.isNotNull() & right_col_str.isNotNull() & (left_col_str == right_col_str)
            )
        else:
            is_equal_col = f.lit(col=False)

        return f.struct(
            left_col.alias("left_value"),
            right_col.alias("right_value"),
            is_equal_col.alias("is_equal"),
            exists_left.alias("exists_left"),
            exists_right.alias("exists_right"),
        ).alias(col_name)

    diff_df = diff.select(
        *[
            comparison_struct(col_name, diff_prefix)
            for col_name, diff_prefix in column_names_diff.items()
            if col_name in diff.columns
        ],
        f.struct(
            f.coalesce(left_df[EXISTS_COL_NAME], f.lit(col=False)).alias("left_value"),
            f.coalesce(right_df[EXISTS_COL_NAME], f.lit(col=False)).alias("right_value"),
        ).alias(EXISTS_COL_NAME),
    )

    diff_df = _add_is_equal_column(diff_df, column_names_diff)
    diff_df = _add_join_cols_column(diff_df, join_cols, granularities)

    return diff_df


def _add_is_equal_column(diff_df: DataFrame, column_names_diff: Dict[str, DiffPrefix]) -> DataFrame:
    """Add the __IS_EQUAL__ column to the diff_df"""
    row_is_equal = f.lit(col=True)
    for col_name, diff_prefix in column_names_diff.items():
        if diff_prefix == DiffPrefix.UNCHANGED and col_name in diff_df.columns:
            row_is_equal = row_is_equal & f.col(f"{col_name}.is_equal")
    diff_df = diff_df.withColumn(IS_EQUAL_COL_NAME, row_is_equal)
    return diff_df


def _get_join_cols_matching_key(join_cols: List[str], key: str) -> List[str]:
    """
    >>> join_cols = ["id", "s__ARRAY____STRUCT__id", "s__ARRAY____STRUCT__ss__ARRAY____STRUCT__id"]
    >>> _get_join_cols_matching_key(join_cols, "")
    ['id']
    >>> _get_join_cols_matching_key(join_cols, "other!")
    []
    >>> _get_join_cols_matching_key(join_cols, "s!")
    ['id', 's__ARRAY____STRUCT__id']
    >>> _get_join_cols_matching_key(join_cols, "s!.ss!")
    ['id', 's__ARRAY____STRUCT__id', 's__ARRAY____STRUCT__ss__ARRAY____STRUCT__id']
    """
    join_cols = [_restore_special_characters(col) for col in join_cols]
    join_cols = [
        col
        for col in join_cols
        if has_same_or_higher_granularity(col, key) and any(has_same_or_higher_granularity(key, c) for c in join_cols)
    ]
    join_cols = [_replace_special_characters(col) for col in join_cols]
    return join_cols


def _build_key_col(join_cols: List[str]) -> Column:
    key_col = f.to_json(
        f.struct([f.coalesce(f.col(col)["left_value"], f.col(col)["right_value"]).alias(col) for col in join_cols]),
    )
    return key_col


def _add_join_cols_column(diff_df: DataFrame, join_cols: List[str], common_keys: List[str]) -> DataFrame:
    """Add the __SAMPLE_ID__ column to the diff_df

    >>> from pyspark.sql import SparkSession
    >>> spark = SparkSession.builder.appName("doctest").getOrCreate()

    >>> diff_df = spark.sql(
    ...     '''
    ...     SELECT INLINE(ARRAY(
    ...         STRUCT(
    ...             STRUCT(1 as left_value, 1 as right_value) as id
    ...         ),
    ...         STRUCT(
    ...             STRUCT(2 as left_value, NULL as right_value) as id
    ...         ),
    ...         STRUCT(
    ...             STRUCT(NULL as left_value, 3 as right_value) as id
    ...         )
    ...     ))
    ... ''',
    ... )
    >>> diff_df.show(truncate=False)
    +---------+
    |id       |
    +---------+
    |{1, 1}   |
    |{2, NULL}|
    |{NULL, 3}|
    +---------+
    <BLANKLINE>
    >>> join_cols = ["id"]
    >>> common_keys = [""]
    >>> _add_join_cols_column(diff_df, join_cols, common_keys).show(truncate=False)
    +---------+-------------+
    |id       |__SAMPLE_ID__|
    +---------+-------------+
    |{1, 1}   |[{"id":1}]   |
    |{2, NULL}|[{"id":2}]   |
    |{NULL, 3}|[{"id":3}]   |
    +---------+-------------+
    <BLANKLINE>

    >>> diff_df = spark.sql(
    ...     '''
    ...     SELECT INLINE(ARRAY(
    ...         STRUCT(
    ...             STRUCT(1 as left_value, 1 as right_value) as id,
    ...             STRUCT(1 as left_value, 1 as right_value) as s__ARRAY____STRUCT__id,
    ...             STRUCT(1 as left_value, 1 as right_value) as s__ARRAY____STRUCT__ss__ARRAY____STRUCT__id
    ...         ),
    ...         STRUCT(
    ...             STRUCT(1 as left_value, 1 as right_value) as id,
    ...             STRUCT(2 as left_value, 2 as right_value) as s__ARRAY____STRUCT__id,
    ...             STRUCT(1 as left_value, 1 as right_value) as s__ARRAY____STRUCT__ss__ARRAY____STRUCT__id
    ...         ),
    ...         STRUCT(
    ...             STRUCT(2 as left_value, 2 as right_value) as id,
    ...             STRUCT(1 as left_value, 1 as right_value) as s__ARRAY____STRUCT__id,
    ...             STRUCT(1 as left_value, 1 as right_value) as s__ARRAY____STRUCT__ss__ARRAY____STRUCT__id
    ...         ),
    ...         STRUCT(
    ...             STRUCT(2 as left_value, 2 as right_value) as id,
    ...             STRUCT(1 as left_value, 1 as right_value) as s__ARRAY____STRUCT__id,
    ...             STRUCT(2 as left_value, 2 as right_value) as s__ARRAY____STRUCT__ss__ARRAY____STRUCT__id
    ...         )
    ...     ))
    ... ''',
    ... )
    >>> diff_df.show(truncate=False)
    +------+----------------------+-------------------------------------------+
    |id    |s__ARRAY____STRUCT__id|s__ARRAY____STRUCT__ss__ARRAY____STRUCT__id|
    +------+----------------------+-------------------------------------------+
    |{1, 1}|{1, 1}                |{1, 1}                                     |
    |{1, 1}|{2, 2}                |{1, 1}                                     |
    |{2, 2}|{1, 1}                |{1, 1}                                     |
    |{2, 2}|{1, 1}                |{2, 2}                                     |
    +------+----------------------+-------------------------------------------+
    <BLANKLINE>
    >>> join_cols = ["id", "s__ARRAY____STRUCT__id", "s__ARRAY____STRUCT__ss__ARRAY____STRUCT__id"]
    >>> common_keys = ["", "other!", "s!", "s!.ss!"]
    >>> _add_join_cols_column(diff_df, join_cols, common_keys).show(truncate=False)
    +------+----------------------+-------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
    |id    |s__ARRAY____STRUCT__id|s__ARRAY____STRUCT__ss__ARRAY____STRUCT__id|__SAMPLE_ID__                                                                                                                           |
    +------+----------------------+-------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
    |{1, 1}|{1, 1}                |{1, 1}                                     |[{"id":1}, {}, {"id":1,"s__ARRAY____STRUCT__id":1}, {"id":1,"s__ARRAY____STRUCT__id":1,"s__ARRAY____STRUCT__ss__ARRAY____STRUCT__id":1}]|
    |{1, 1}|{2, 2}                |{1, 1}                                     |[{"id":1}, {}, {"id":1,"s__ARRAY____STRUCT__id":2}, {"id":1,"s__ARRAY____STRUCT__id":2,"s__ARRAY____STRUCT__ss__ARRAY____STRUCT__id":1}]|
    |{2, 2}|{1, 1}                |{1, 1}                                     |[{"id":2}, {}, {"id":2,"s__ARRAY____STRUCT__id":1}, {"id":2,"s__ARRAY____STRUCT__id":1,"s__ARRAY____STRUCT__ss__ARRAY____STRUCT__id":1}]|
    |{2, 2}|{1, 1}                |{2, 2}                                     |[{"id":2}, {}, {"id":2,"s__ARRAY____STRUCT__id":1}, {"id":2,"s__ARRAY____STRUCT__id":1,"s__ARRAY____STRUCT__ss__ARRAY____STRUCT__id":2}]|
    +------+----------------------+-------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
    <BLANKLINE>
    """  # noqa: E501
    key_cols = [_build_key_col(_get_join_cols_matching_key(join_cols, key)) for key in common_keys]
    diff_df = diff_df.withColumn(SAMPLE_ID_COL_NAME, f.array(key_cols))
    return diff_df


def _harmonize_and_normalize_dataframes(
    left_flat: DataFrame,
    right_flat: DataFrame,
    skip_make_dataframes_comparable: bool,
) -> Tuple[DataFrame, DataFrame]:
    if not skip_make_dataframes_comparable:
        left_flat, right_flat = harmonize_dataframes(
            left_flat,
            right_flat,
            keep_missing_columns=True,
        )
    left_flat = sort_all_arrays(left_flat)
    right_flat = sort_all_arrays(right_flat)
    return left_flat, right_flat


def _build_diff_dataframe_shards(
    left_df: DataFrame,
    right_df: DataFrame,
    schema_diff_result: SchemaDiffResult,
    join_cols: List[str],
    specified_join_cols: Optional[List[str]],
) -> Dict[str, DataFrame]:
    left_fields_to_unnest = [
        col_name
        for col_name, diff_prefix in schema_diff_result.column_names_diff.items()
        if diff_prefix in [DiffPrefix.UNCHANGED, DiffPrefix.REMOVED]
    ]
    right_fields_to_unnest = [
        col_name
        for col_name, diff_prefix in schema_diff_result.column_names_diff.items()
        if diff_prefix in [DiffPrefix.UNCHANGED, DiffPrefix.ADDED]
    ]

    unnested_left_dfs = unnest_fields(
        left_df,
        left_fields_to_unnest,
        keep_fields=join_cols,
    )
    unnested_right_dfs = unnest_fields(
        right_df,
        right_fields_to_unnest,
        keep_fields=join_cols,
    )

    granularities = sorted(
        set(unnested_left_dfs.keys()).intersection(set(unnested_right_dfs.keys())),
    )

    def build_shard(granularity: str) -> DataFrame:
        l_df = unnested_left_dfs[granularity]
        r_df = unnested_right_dfs[granularity]
        l_df = _replace_special_characters_from_col_names(l_df)
        r_df = _replace_special_characters_from_col_names(r_df)
        schema_diff_result = diff_dataframe_schemas(l_df, r_df, join_cols)
        new_join_cols = [_replace_special_characters(col) for col in join_cols]
        new_join_cols = [col for col in new_join_cols if col in l_df.columns]
        new_join_cols, self_join_growth_estimate = _get_join_cols(
            l_df,
            r_df,
            new_join_cols,
        )
        _check_join_cols(specified_join_cols, new_join_cols, self_join_growth_estimate)
        diff_df = _build_diff_dataframe(l_df, r_df, schema_diff_result.column_names_diff, new_join_cols, granularities)
        return diff_df

    return {granularity: build_shard(granularity) for granularity in granularities}


def compare_dataframes(
    left_df: DataFrame,
    right_df: DataFrame,
    join_cols: Optional[List[str]] = None,
) -> DiffResult:
    """Compares two DataFrames and return a [`DiffResult`][spark_frame.data_diff.diff_result.DiffResult] object.

    We first compare the DataFrame schemas. If the schemas are different, we adapt the DataFrames to make them
    as much comparable as possible:
    - If the order of the columns changed, we re-order them automatically to perform the diff
    - If the order of the fields inside a struct changed, we re-order them automatically to perform the diff
    - If a column type changed, we cast the column to the smallest common type
    - We don't recognize when a column is renamed, we treat it as if the old column was removed and the new column added

    If `join_cols` is specified, we will use the specified columns to perform the comparison join between the
    two DataFrames. Ideally, the `join_cols` should respect an unicity constraint.

    If they contain duplicates, a safety check is performed to prevent a potential combinatorial explosion:
    if the number of rows in the joined DataFrame would be more than twice the size of the original DataFrames,
    then an Exception is raised and the user will be asked to provide another set of `join_cols`.

    If no `join_cols` is specified, the algorithm will try to automatically find a single column suitable for
    the join. However, the automatic inference can only find join keys based on a single column.
    If the DataFrame's unique keys are composite (multiple columns) they must be given explicitly via `join_cols`
    to perform the diff analysis.

    !!! tip "Tips"
        - If you want to test a column renaming, you can temporarily add renaming steps to the DataFrame
          you want to test.
        - If you want to exclude columns from the diff, you can simply drop them from the DataFrames you want to
          compare.
        - When comparing arrays, this algorithm ignores their ordering (e.g. `[1, 2, 3] == [3, 2, 1]`).
        - When dealing with a nested structure, if the struct contains a unique identifier, it can be specified
          in the join_cols and the structure will be automatically unnested in the diff results.
          For instance, if we have a structure `my_array: ARRAY<STRUCT<a, b, ...>>`
          and if `a` is a unique identifier, then you can add `"my_array!.a"` in the join_cols argument.
          (cf. Example 2)

    Args:
        left_df: A Spark DataFrame
        right_df: Another DataFrame
        join_cols: Specifies the columns on which the two DataFrames should be joined to compare them

    Returns:
        A DiffResult object

    Examples: Example 1: simple diff
        >>> from spark_frame.data_diff.compare_dataframes_impl import __get_test_dfs
        >>> from spark_frame.data_diff import compare_dataframes
        >>> df1, df2 = __get_test_dfs()

        >>> df1.show()
        +---+-----------+
        | id|   my_array|
        +---+-----------+
        |  1|[{1, 2, 3}]|
        |  2|[{1, 2, 3}]|
        |  3|[{1, 2, 3}]|
        +---+-----------+
        <BLANKLINE>

        >>> df2.show()
        +---+--------------+
        | id|      my_array|
        +---+--------------+
        |  1|[{1, 2, 3, 4}]|
        |  2|[{2, 2, 3, 4}]|
        |  4|[{1, 2, 3, 4}]|
        +---+--------------+
        <BLANKLINE>

        >>> diff_result = compare_dataframes(df1, df2)
        <BLANKLINE>
        Analyzing differences...
        No join_cols provided: trying to automatically infer a column that can be used for joining the two DataFrames
        Found the following column: id
        Generating the diff by joining the DataFrames together using the inferred column: id

        >>> diff_result.display()
        Schema has changed:
        @@ -1,2 +1,2 @@
        <BLANKLINE>
         id INT
        -my_array ARRAY<STRUCT<a:INT,b:INT,c:INT>>
        +my_array ARRAY<STRUCT<a:INT,b:INT,c:INT,d:INT>>
        WARNING: columns that do not match both sides will be ignored
        <BLANKLINE>
        diff NOT ok
        <BLANKLINE>
        Row count ok: 3 rows
        <BLANKLINE>
        0 (0.0%) rows are identical
        2 (50.0%) rows have changed
        1 (25.0%) rows are only in 'left'
        1 (25.0%) rows are only in 'right
        <BLANKLINE>
        Found the following changes:
        +-----------+-------------+---------------------+---------------------------+--------------+
        |column_name|total_nb_diff|left_value           |right_value                |nb_differences|
        +-----------+-------------+---------------------+---------------------------+--------------+
        |my_array   |2            |[{"a":1,"b":2,"c":3}]|[{"a":1,"b":2,"c":3,"d":4}]|1             |
        |my_array   |2            |[{"a":1,"b":2,"c":3}]|[{"a":2,"b":2,"c":3,"d":4}]|1             |
        +-----------+-------------+---------------------+---------------------------+--------------+
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
        +-----------+---------------------------+---+
        |column_name|value                      |nb |
        +-----------+---------------------------+---+
        |id         |4                          |1  |
        |my_array   |[{"a":1,"b":2,"c":3,"d":4}]|1  |
        +-----------+---------------------------+---+
        <BLANKLINE>

        >>> diff_result.export_to_html(output_file_path="test_working_dir/compare_dataframes_example_1.html")
        Report exported as test_working_dir/compare_dataframes_example_1.html

        [Check out the exported report here](../diff_reports/compare_dataframes_example_1.html)

    Examples: Example 2: diff on complex structures
        By adding `"my_array!.a"` to the join_cols argument, the array gets unnested for the diff
        >>> diff_result_unnested = compare_dataframes(df1, df2, join_cols=["id", "my_array!.a"])
        <BLANKLINE>
        Analyzing differences...
        Generating the diff by joining the DataFrames together using the provided column: id
        Generating the diff by joining the DataFrames together using the provided columns: ['id', 'my_array!.a']

        >>> diff_result_unnested.display()
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
        WARNING: This diff has multiple granularity levels, we will print the results for each granularity level,
                 but we recommend to export the results to html for a much more digest result.
        <BLANKLINE>
        ##############################################################
        Granularity : root (4 rows)
        <BLANKLINE>
        Row count ok: 3 rows
        <BLANKLINE>
        2 (50.0%) rows are identical
        0 (0.0%) rows have changed
        1 (25.0%) rows are only in 'left'
        1 (25.0%) rows are only in 'right
        <BLANKLINE>
        1 rows were only found in 'left' :
        Most frequent values in 'left' for each column :
        +-----------+-----+---+
        |column_name|value|nb |
        +-----------+-----+---+
        |id         |3    |1  |
        |my_array!.a|1    |2  |
        |my_array!.b|2    |2  |
        |my_array!.c|3    |2  |
        +-----------+-----+---+
        <BLANKLINE>
        1 rows were only found in 'right' :
        Most frequent values in 'right' for each column :
        +-----------+-----+---+
        |column_name|value|nb |
        +-----------+-----+---+
        |id         |4    |1  |
        |my_array!.a|1    |1  |
        |my_array!.a|2    |1  |
        |my_array!.b|2    |2  |
        |my_array!.c|3    |2  |
        |my_array!.d|4    |3  |
        +-----------+-----+---+
        <BLANKLINE>
        ##############################################################
        Granularity : my_array! (5 rows)
        <BLANKLINE>
        Row count ok: 3 rows
        <BLANKLINE>
        1 (20.0%) rows are identical
        0 (0.0%) rows have changed
        2 (40.0%) rows are only in 'left'
        2 (40.0%) rows are only in 'right
        <BLANKLINE>
        2 rows were only found in 'left' :
        Most frequent values in 'left' for each column :
        +-----------+-----+---+
        |column_name|value|nb |
        +-----------+-----+---+
        |id         |3    |1  |
        |my_array!.a|1    |2  |
        |my_array!.b|2    |2  |
        |my_array!.c|3    |2  |
        +-----------+-----+---+
        <BLANKLINE>
        2 rows were only found in 'right' :
        Most frequent values in 'right' for each column :
        +-----------+-----+---+
        |column_name|value|nb |
        +-----------+-----+---+
        |id         |4    |1  |
        |my_array!.a|1    |1  |
        |my_array!.a|2    |1  |
        |my_array!.b|2    |2  |
        |my_array!.c|3    |2  |
        |my_array!.d|4    |3  |
        +-----------+-----+---+
        <BLANKLINE>

        >>> diff_result_unnested.export_to_html(output_file_path="test_working_dir/compare_dataframes_example_2.html")
        Report exported as test_working_dir/compare_dataframes_example_2.html

        [Check out the exported report here](../diff_reports/compare_dataframes_example_2.html)
    """
    print("\nAnalyzing differences...")

    if join_cols == []:
        join_cols = None
    specified_join_cols = join_cols
    left_df = convert_all_maps_to_arrays(left_df)
    right_df = convert_all_maps_to_arrays(right_df)

    if join_cols is None:
        left_flat = flatten(left_df, struct_separator=STRUCT_SEPARATOR_REPLACEMENT)
        right_flat = flatten(right_df, struct_separator=STRUCT_SEPARATOR_REPLACEMENT)
        join_cols, _ = _get_join_cols(
            left_flat,
            right_flat,
            join_cols,
        )
    else:
        validate_fields_exist(join_cols, nested.fields(left_df))
        validate_fields_exist(join_cols, nested.fields(right_df))

    global_schema_diff_result = diff_dataframe_schemas(left_df, right_df, join_cols)
    left_df, right_df = _harmonize_and_normalize_dataframes(
        left_df,
        right_df,
        skip_make_dataframes_comparable=global_schema_diff_result.same_schema,
    )

    diff_dataframe_shards = _build_diff_dataframe_shards(
        left_df,
        right_df,
        global_schema_diff_result,
        join_cols,
        specified_join_cols,
    )
    diff_result = DiffResult(
        global_schema_diff_result,
        diff_dataframe_shards,
        join_cols,
    )

    return diff_result


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
    """,
    )
    df2 = spark.sql(
        """
        SELECT INLINE (ARRAY(
            STRUCT(1 as id, ARRAY(STRUCT(1 as a, 2 as b, 3 as c, 4 as d)) as my_array),
            STRUCT(2 as id, ARRAY(STRUCT(2 as a, 2 as b, 3 as c, 4 as d)) as my_array),
            STRUCT(4 as id, ARRAY(STRUCT(1 as a, 2 as b, 3 as c, 4 as d)) as my_array)
       ))
    """,
    )
    return df1, df2
