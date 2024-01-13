from pyspark.sql import SparkSession

from spark_frame import nested
from spark_frame.utils import show_string, strip_margin


def test_unnest_all_fields_with_fields_having_same_name_inside_structs(spark: SparkSession):
    """
    GIVEN a DataFrame with nested fields having the same field name
    WHEN I call unnest_all_fields on it
    THEN the result should be correct
    """
    df = spark.sql(
        """
        SELECT
            1 as id,
            STRUCT(2 as id) as s1,
            ARRAY(STRUCT(3 as id, ARRAY(STRUCT(5 as id), STRUCT(6 as id)) as s3)) as s2
    """,
    )
    assert show_string(df) == strip_margin(
        """
        |+---+---+-----------------+
        || id| s1|               s2|
        |+---+---+-----------------+
        ||  1|{2}|[{3, [{5}, {6}]}]|
        |+---+---+-----------------+
        |""",
    )
    assert nested.fields(df) == ["id", "s1.id", "s2!.id", "s2!.s3!.id"]
    result_df_list = nested.unnest_all_fields(df, keep_columns=["id"])
    assert list(result_df_list.keys()) == ["", "s2", "s2!.s3"]
    assert show_string(result_df_list[""]) == strip_margin(
        """
        |+---+-----+
        || id|s1.id|
        |+---+-----+
        ||  1|    2|
        |+---+-----+
        |""",
    )
    assert show_string(result_df_list["s2"]) == strip_margin(
        """
        |+---+------+
        || id|s2!.id|
        |+---+------+
        ||  1|     3|
        |+---+------+
        |""",
    )
    assert show_string(result_df_list["s2!.s3"]) == strip_margin(
        """
        |+---+----------+
        || id|s2!.s3!.id|
        |+---+----------+
        ||  1|         5|
        ||  1|         6|
        |+---+----------+
        |""",
    )


def test_unnest_fields_with_fields_having_same_name_inside_array_structs(spark: SparkSession):
    """
    GIVEN a DataFrame with fields in array of struct having the same name as root-level columns
    WHEN we apply unnest_fields on it
    THEN the result should be correct
    """
    df = spark.sql(
        """
        SELECT
            1 as id,
            STRUCT(2 as id) as s1,
            ARRAY(STRUCT(3 as id, ARRAY(STRUCT(5 as id), STRUCT(6 as id)) as s3)) as s2
    """,
    )
    assert show_string(df) == strip_margin(
        """
        |+---+---+-----------------+
        || id| s1|               s2|
        |+---+---+-----------------+
        ||  1|{2}|[{3, [{5}, {6}]}]|
        |+---+---+-----------------+
        |""",
    )

    assert nested.fields(df) == ["id", "s1.id", "s2!.id", "s2!.s3!.id"]
    result_df_list = nested.unnest_all_fields(df, keep_columns=["id"])
    assert show_string(result_df_list[""]) == strip_margin(
        """
        |+---+-----+
        || id|s1.id|
        |+---+-----+
        ||  1|    2|
        |+---+-----+
        |""",
    )
    assert show_string(result_df_list["s2"]) == strip_margin(
        """
        |+---+------+
        || id|s2!.id|
        |+---+------+
        ||  1|     3|
        |+---+------+
        |""",
    )
    assert show_string(result_df_list["s2!.s3"]) == strip_margin(
        """
        |+---+----------+
        || id|s2!.s3!.id|
        |+---+----------+
        ||  1|         5|
        ||  1|         6|
        |+---+----------+
        |""",
    )
