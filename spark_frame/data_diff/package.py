from functools import lru_cache
from typing import TypeVar

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import StructField

from spark_frame.data_type_utils import is_repeated

EXISTS_COL_NAME = "__EXISTS__"
IS_EQUAL_COL_NAME = "__IS_EQUAL__"
SAMPLE_ID_COL_NAME = "__SAMPLE_ID__"
STRUCT_SEPARATOR_REPLACEMENT = "__STRUCT__"
REPETITION_MARKER_REPLACEMENT = "__ARRAY__"

A = TypeVar("A")
B = TypeVar("B")


class Predicates:
    # These predicates must be defined in a lazy way, because they require the SparkSession to be instatiated.

    @property
    def present_in_both(self) -> Column:
        return f.col(f"{EXISTS_COL_NAME}.left_value") & f.col(
            f"{EXISTS_COL_NAME}.right_value",
        )

    @property
    def in_left(self) -> Column:
        return f.col(f"{EXISTS_COL_NAME}.left_value")

    @property
    def in_right(self) -> Column:
        return f.col(f"{EXISTS_COL_NAME}.right_value")

    @property
    def only_in_left(self) -> Column:
        return f.col(f"{EXISTS_COL_NAME}.left_value") & (f.col(f"{EXISTS_COL_NAME}.right_value") == f.lit(col=False))

    @property
    def only_in_right(self) -> Column:
        return (f.col(f"{EXISTS_COL_NAME}.left_value") == f.lit(col=False)) & f.col(
            f"{EXISTS_COL_NAME}.right_value",
        )

    @property
    def row_is_equal(self) -> Column:
        return f.col(IS_EQUAL_COL_NAME)

    @property
    def row_changed(self) -> Column:
        return f.col(IS_EQUAL_COL_NAME) == f.lit(col=False)


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
    return col.cast("STRING")


@lru_cache
def _get_test_diff_df() -> DataFrame:
    """
    >>> from spark_frame.data_diff.package import _get_test_diff_df
    >>> _get_test_diff_df().show(truncate=False)
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
    """  # noqa: E501
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("doctest").getOrCreate()
    diff_df = spark.sql(
        """
        SELECT INLINE(ARRAY(
            STRUCT(
                STRUCT(1 as left_value, 1 as right_value, True as is_equal,
                       True as exists_left, True as exists_right
                ) as id,
                STRUCT("a" as left_value, "a" as right_value, True as is_equal,
                       True as exists_left, True as exists_right
                ) as c1,
                STRUCT(1 as left_value, 1 as right_value, True as is_equal,
                       True as exists_left, True as exists_right
                ) as c2,
                STRUCT(1 as left_value, CAST(NULL as INT) as right_value, False as is_equal,
                       True as exists_left, False as exists_right
                ) as c3,
                STRUCT(CAST(NULL as INT) as left_value, 1 as right_value, False as is_equal,
                       False as exists_left, True as exists_right
                ) as c4,
                STRUCT(True as left_value, True as right_value) as __EXISTS__,
                TRUE as __IS_EQUAL__,
                ARRAY('{"id": 1}') as __SAMPLE_ID__
            ),
            STRUCT(
                STRUCT(2 as left_value, 2 as right_value, True as is_equal,
                       True as exists_left, True as exists_right
                ) as id,
                STRUCT("b" as left_value, "b" as right_value, True as is_equal,
                       True as exists_left, True as exists_right
                ) as c1,
                STRUCT(2 as left_value, 3 as right_value, False as is_equal,
                       True as exists_left, True as exists_right
                ) as c2,
                STRUCT(1 as left_value, CAST(NULL as INT) as right_value, False as is_equal,
                       True as exists_left, False as exists_right
                ) as c3,
                STRUCT(CAST(NULL as INT) as left_value, 1 as right_value, False as is_equal,
                       False as exists_left, True as exists_right
                ) as c4,
                STRUCT(True as left_value, True as right_value) as __EXISTS__,
                FALSE as __IS_EQUAL__,
                ARRAY('{"id": 2}') as __SAMPLE_ID__
            ),
            STRUCT(
                STRUCT(3 as left_value, 3 as right_value, True as is_equal,
                       True as exists_left, True as exists_right
                ) as id,
                STRUCT("b" as left_value, "b" as right_value, True as is_equal,
                       True as exists_left, True as exists_right
                ) as c1,
                STRUCT(2 as left_value, 4 as right_value, False as is_equal,
                       True as exists_left, True as exists_right
                ) as c2,
                STRUCT(2 as left_value, CAST(NULL as INT) as right_value, False as is_equal,
                       True as exists_left, False as exists_right
                ) as c3,
                STRUCT(CAST(NULL as INT) as left_value, 2 as right_value, False as is_equal,
                       False as exists_left, True as exists_right
                ) as c4,
                STRUCT(True as left_value, True as right_value) as __EXISTS__,
                FALSE as __IS_EQUAL__,
                ARRAY('{"id": 3}') as __SAMPLE_ID__
            ),
            STRUCT(
                STRUCT(4 as left_value, 4 as right_value, True as is_equal,
                       True as exists_left, True as exists_right
                ) as id,
                STRUCT("b" as left_value, "b" as right_value, True as is_equal,
                       True as exists_left, True as exists_right
                ) as c1,
                STRUCT(2 as left_value, 4 as right_value, False as is_equal,
                       True as exists_left, True as exists_right
                ) as c2,
                STRUCT(2 as left_value, CAST(NULL as INT) as right_value, False as is_equal,
                       True as exists_left, False as exists_right
                ) as c3,
                STRUCT(CAST(NULL as INT) as left_value, 2 as right_value, False as is_equal,
                       False as exists_left, True as exists_right
                ) as c4,
                STRUCT(True as left_value, True as right_value) as __EXISTS__,
                FALSE as __IS_EQUAL__,
                ARRAY('{"id": 4}') as __SAMPLE_ID__
            ),
            STRUCT(
                STRUCT(5 as left_value, CAST(NULL as INT) as right_value, False as is_equal,
                       True as exists_left, False as exists_right
                ) as id,
                STRUCT("c" as left_value, CAST(NULL as INT) as right_value, False as is_equal,
                       True as exists_left, False as exists_right
                ) as c1,
                STRUCT(3 as left_value, CAST(NULL as INT) as right_value, False as is_equal,
                       True as exists_left, False as exists_right
                ) as c2,
                STRUCT(3 as left_value, CAST(NULL as INT) as right_value, False as is_equal,
                       True as exists_left, False as exists_right
                ) as c3,
                STRUCT(CAST(NULL as INT) as left_value, CAST(NULL as INT) as right_value, False as is_equal,
                       False as exists_left, False as exists_right
                ) as c4,
                STRUCT(True as left_value, False as right_value) as __EXISTS__,
                FALSE as __IS_EQUAL__,
                ARRAY('{"id": 5}') as __SAMPLE_ID__
            ),
            STRUCT(
                STRUCT(CAST(NULL as INT) as left_value, 6 as right_value, False as is_equal,
                       False as exists_left, True as exists_right
                ) as id,
                STRUCT(CAST(NULL as INT) as left_value, "f" as right_value, False as is_equal,
                       False as exists_left, True as exists_right
                ) as c1,
                STRUCT(CAST(NULL as INT) as left_value, 3 as right_value, False as is_equal,
                       False as exists_left, True as exists_right
                ) as c2,
                STRUCT(CAST(NULL as INT) as left_value, CAST(NULL as INT) as right_value, False as is_equal,
                       False as exists_left, False as exists_right
                ) as c3,
                STRUCT(CAST(NULL as INT) as left_value, 3 as right_value, False as is_equal,
                       False as exists_left, True as exists_right
                ) as c4,
                STRUCT(False as left_value, True as right_value) as __EXISTS__,
                FALSE as __IS_EQUAL__,
                ARRAY('{"id": 6}') as __SAMPLE_ID__
            )
        ))
    """,
    )
    return diff_df
