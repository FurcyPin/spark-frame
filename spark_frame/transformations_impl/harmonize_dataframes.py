from typing import Dict, Optional, Tuple

from pyspark.sql import DataFrame

from spark_frame import fp
from spark_frame.data_type_utils import flatten_schema, get_common_columns
from spark_frame.fp import higher_order
from spark_frame.fp.printable_function import PrintableFunction
from spark_frame.nested_impl.package import (
    _build_nested_struct_tree,
    _build_transformation_from_tree,
    _deepest_granularity,
)


def harmonize_dataframes(
    left_df: DataFrame,
    right_df: DataFrame,
    common_columns: Optional[Dict[str, Optional[str]]] = None,
    keep_missing_columns: bool = False,
) -> Tuple[DataFrame, DataFrame]:
    """Given two DataFrames, returns two new corresponding DataFrames with the same schemas by applying the following
    changes:

    - Only common columns are kept
    - Columns of type MAP<key, value> are cast into ARRAY<STRUCT<key, value>>
    - Columns are re-ordered to have the same ordering in both DataFrames
    - When matching columns have different types, their type is widened to their most narrow common type.
    This transformation is applied recursively on nested columns, including those inside
    repeated records (a.k.a. ARRAY<STRUCT<>>).

    Args:
        left_df: A Spark DataFrame
        right_df: A Spark DataFrame
        common_columns: A dict of (column name, type).
            Column names must appear in both DataFrames, and each column will be cast into the corresponding type.
        keep_missing_columns: If set to true, the root columns of each DataFrames that do not exist in the other
            one are kept.

    Returns:
        Two new Spark DataFrames with the same schema

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df1 = spark.sql('SELECT 1 as id, STRUCT(2 as a, ARRAY(STRUCT(3 as b, 4 as c)) as s2) as s1')
        >>> df2 = spark.sql('SELECT 1 as id, STRUCT(2 as a, ARRAY(STRUCT(3.0 as b, "4" as c, 5 as d)) as s2) as s1')
        >>> df1.union(df2).show(truncate=False) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        AnalysisException: ... UNION can only be performed on tables with compatible column types.
        >>> df1, df2 = harmonize_dataframes(df1, df2)
        >>> df1.union(df2).show()
        +---+---------------+
        | id|             s1|
        +---+---------------+
        |  1|{2, [{3.0, 4}]}|
        |  1|{2, [{3.0, 4}]}|
        +---+---------------+
        <BLANKLINE>
        >>> df1, df2 = harmonize_dataframes(df1, df2, common_columns={"id": None, "s1.s2!.b": "int"})
        >>> df1.union(df2).show()
        +---+-------+
        | id|     s1|
        +---+-------+
        |  1|{[{3}]}|
        |  1|{[{3}]}|
        +---+-------+
        <BLANKLINE>
    """
    left_schema_flat = flatten_schema(left_df.schema, explode=True)
    right_schema_flat = flatten_schema(right_df.schema, explode=True)
    if common_columns is None:
        common_columns = get_common_columns(left_schema_flat, right_schema_flat)

    left_only_columns = {}
    right_only_columns = {}
    if keep_missing_columns:
        left_cols = [field.name for field in left_schema_flat.fields]
        right_cols = [field.name for field in right_schema_flat.fields]
        left_cols_set = set(left_cols)
        right_cols_set = set(right_cols)
        left_only_columns = {col: None for col in left_cols if col not in right_cols_set}
        right_only_columns = {col: None for col in right_cols if col not in left_cols_set}

    def build_col(col_name: str, col_type: Optional[str]) -> PrintableFunction:
        parent_structs = _deepest_granularity(col_name)
        if col_type is not None:
            tpe = col_type
            f1 = PrintableFunction(lambda s: s.cast(tpe), lambda s: f"{s}.cast({tpe})")
        else:
            f1 = higher_order.identity
        f2 = higher_order.recursive_struct_get(parent_structs)
        return fp.compose(f1, f2)

    left_columns = {**common_columns, **left_only_columns}
    right_columns = {**common_columns, **right_only_columns}
    left_columns_dict = {col_name: build_col(col_name, col_type) for (col_name, col_type) in left_columns.items()}
    right_columns_dict = {col_name: build_col(col_name, col_type) for (col_name, col_type) in right_columns.items()}
    left_tree = _build_nested_struct_tree(left_columns_dict)
    right_tree = _build_nested_struct_tree(right_columns_dict)
    left_root_transformation = _build_transformation_from_tree(left_tree)
    right_root_transformation = _build_transformation_from_tree(right_tree)
    return (
        left_df.select(*left_root_transformation([left_df])),
        right_df.select(*right_root_transformation([right_df])),
    )
