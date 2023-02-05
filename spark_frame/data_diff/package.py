from typing import TypeVar

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import StructField

from spark_frame.data_type_utils import is_repeated

MAGIC_HASH_COL_NAME = "__MAGIC_HASH__"
EXISTS_COL_NAME = "__EXISTS__"
IS_EQUAL_COL_NAME = "__IS_EQUAL__"
STRUCT_SEPARATOR_REPLACEMENT = "__STRUCT__"

A = TypeVar("A")
B = TypeVar("B")


class Predicates:
    # These predicates must be defined in a lazy way, because they require the SparkSession to be instatiated.

    @property
    def present_in_both(self):
        return f.col(f"{EXISTS_COL_NAME}.left_value") & f.col(f"{EXISTS_COL_NAME}.right_value")

    @property
    def in_left(self):
        return f.col(f"{EXISTS_COL_NAME}.left_value")

    @property
    def in_right(self):
        return f.col(f"{EXISTS_COL_NAME}.right_value")

    @property
    def only_in_left(self):
        return f.col(f"{EXISTS_COL_NAME}.left_value") & (f.col(f"{EXISTS_COL_NAME}.right_value") == f.lit(False))

    @property
    def only_in_right(self):
        return (f.col(f"{EXISTS_COL_NAME}.left_value") == f.lit(False)) & f.col(f"{EXISTS_COL_NAME}.right_value")

    @property
    def row_is_equal(self):
        return f.col(IS_EQUAL_COL_NAME)

    @property
    def row_changed(self):
        return f.col(IS_EQUAL_COL_NAME) == f.lit(False)


PREDICATES = Predicates()


def canonize_col(col: Column, schema_field: StructField) -> Column:
    """Applies a transformation on the specified field depending on its type.
    This ensures that the hash of the column will be constant despite it's ordering.

    :param col: the SchemaField object
    :param schema_field: the parent DataFrame
    :return: a Column
    """
    if is_repeated(schema_field):
        col = f.when(~col.isNull(), f.to_json(col))
    return col


def _get_test_diff_df() -> DataFrame:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("doctest").getOrCreate()
    diff_df = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(
                1 as id,
                STRUCT("a" as left_value, "a" as right_value, True as is_equal) as c1,
                STRUCT(1 as left_value, 1 as right_value, True as is_equal) as c2,
                STRUCT(True as left_value, True as right_value) as __EXISTS__,
                TRUE as __IS_EQUAL__
            ),
            STRUCT(
                2 as id,
                STRUCT("b" as left_value, "b" as right_value, True as is_equal) as c1,
                STRUCT(2 as left_value, 3 as right_value, False as is_equal) as c2,
                STRUCT(True as left_value, True as right_value) as __EXISTS__,
                FALSE as __IS_EQUAL__
            ),
            STRUCT(
                3 as id,
                STRUCT("b" as left_value, "b" as right_value, True as is_equal) as c1,
                STRUCT(2 as left_value, 4 as right_value, False as is_equal) as c2,
                STRUCT(True as left_value, True as right_value) as __EXISTS__,
                FALSE as __IS_EQUAL__
            ),
            STRUCT(
                4 as id,
                STRUCT("b" as left_value, "b" as right_value, True as is_equal) as c1,
                STRUCT(2 as left_value, 4 as right_value, False as is_equal) as c2,
                STRUCT(True as left_value, True as right_value) as __EXISTS__,
                FALSE as __IS_EQUAL__
            ),
            STRUCT(
                5 as id,
                STRUCT("c" as left_value, NULL as right_value, False as is_equal) as c1,
                STRUCT(3 as left_value, NULL as right_value, False as is_equal) as c2,
                STRUCT(True as left_value, False as right_value) as __EXISTS__,
                FALSE as __IS_EQUAL__
            ),
            STRUCT(
                6 as id,
                STRUCT(NULL as left_value, "f" as right_value, False as is_equal) as c1,
                STRUCT(NULL as left_value, 3 as right_value, False as is_equal) as c2,
                STRUCT(False as left_value, True as right_value) as __EXISTS__,
                FALSE as __IS_EQUAL__
            )
        ))
    """
    )
    return diff_df
