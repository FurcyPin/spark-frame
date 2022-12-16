from typing import Optional, Sequence, Tuple, cast

from pyspark.sql import DataFrame

from spark_frame import fp
from spark_frame.conf import REPETITION_MARKER, STRUCT_SEPARATOR
from spark_frame.data_type_utils import flatten_schema, get_common_columns
from spark_frame.fp import higher_order
from spark_frame.fp.printable_function import PrintableFunction
from spark_frame.nested import resolve_nested_columns


def harmonize_dataframes(
    left_df: DataFrame, right_df: DataFrame, common_columns: Optional[Sequence[Tuple[str, Optional[str]]]] = None
) -> Tuple[DataFrame, DataFrame]:
    """Given two DataFrames, returns two new corresponding DataFrames with the same schemas by applying the following
    changes:
    - Only common columns are kept
    - Columns of type MAP<key, value> are cast into ARRAY<STRUCT<key, value>>
    - Columns are re-order to have the same ordering in both DataFrames
    - When matching columns have different types, their type is widened to their most narrow common type.
    This transformation is applied recursively on nested columns, including those inside
    repeated records (a.k.a. ARRAY<STRUCT<>>).

    Args:
        left_df: A Spark DataFrame
        right_df: A Spark DataFrame
        common_columns: A list of (column name, type) tuples.
            Column names must appear in both DataFrames, and each column will be cast into the corresponding type.

    Returns:
        Two new Spark DataFrames with the same schema

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df1 = spark.sql('SELECT 1 as id, STRUCT(2 as a, ARRAY(STRUCT(3 as b, 4 as c)) as s2) as s1')
        >>> df2 = spark.sql('SELECT 1 as id, STRUCT(2 as a, ARRAY(STRUCT(3.0 as b, "4" as c, 5 as d)) as s2) as s1')
        >>> df1.union(df2).show(truncate=False) # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        pyspark.sql.utils.AnalysisException: Union can only be performed on tables with the compatible column types. ...
        >>> df1, df2 = harmonize_dataframes(df1, df2)
        >>> df1.union(df2).show()
        +---+---------------+
        | id|             s1|
        +---+---------------+
        |  1|{2, [{3.0, 4}]}|
        |  1|{2, [{3.0, 4}]}|
        +---+---------------+
        <BLANKLINE>
    """
    if common_columns is None:
        left_schema_flat = flatten_schema(left_df.schema, explode=True)
        right_schema_flat = flatten_schema(right_df.schema, explode=True)
        common_columns = get_common_columns(left_schema_flat, right_schema_flat)

    def build_col(col_name: str, col_type: Optional[str]) -> PrintableFunction:
        is_repeated = col_name[-1] == REPETITION_MARKER
        col = col_name.split(STRUCT_SEPARATOR)[-1]
        if col_type is not None:
            tpe = cast(str, col_type)
            f1 = PrintableFunction(lambda s: s.cast(tpe), lambda s: f"{s}.cast({tpe})")
        else:
            f1 = higher_order.identity
        if is_repeated:
            f2 = higher_order.identity
        else:
            f2 = higher_order.safe_struct_get(col)
        return fp.compose(f1, f2)

    common_columns_dict = {col_name: build_col(col_name, col_type) for (col_name, col_type) in common_columns}
    resolved_columns = resolve_nested_columns(common_columns_dict)
    return left_df.select(*resolved_columns), right_df.select(*resolved_columns)
