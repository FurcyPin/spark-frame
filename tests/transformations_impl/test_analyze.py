from pyspark.sql import SparkSession

from spark_frame.transformations_impl.analyze import __get_test_df, analyze
from spark_frame.utils import show_string, strip_margin

field_to_index = {
    "column_name": 0,
    "column_type": 1,
    "count": 2,
    "count_distinct": 3,
    "count_null": 4,
    "min": 5,
    "max": 6,
    "approx_top_100": 7,
}


def test_analyze():
    df = __get_test_df()
    actual = analyze(df)
    assert show_string(actual, truncate=False) == strip_margin(
        """
        |+-------------+----------------------+-----------+-----+--------------+----------+---------+---------+
        ||column_number|column_name           |column_type|count|count_distinct|count_null|min      |max      |
        |+-------------+----------------------+-----------+-----+--------------+----------+---------+---------+
        ||0            |id                    |INTEGER    |9    |9             |0         |1        |9        |
        ||1            |name                  |STRING     |9    |9             |0         |Blastoise|Wartortle|
        ||2            |types!                |STRING     |13   |5             |0         |Fire     |Water    |
        ||3            |evolution.can_evolve  |BOOLEAN    |9    |2             |0         |false    |true     |
        ||4            |evolution.evolves_from|INTEGER    |9    |6             |3         |1        |8        |
        |+-------------+----------------------+-----------+-----+--------------+----------+---------+---------+
        |""",
    )


def test_analyze_with_keyword_column_names(spark: SparkSession):
    """Analyze method should still work on DataFrames with columns names that collision with SQL keywords
    such as 'FROM'."""
    query = """SELECT 1 as `FROM`, STRUCT('a' as `ALL`) as `UNION`"""
    df = spark.sql(query)
    actual = analyze(df)
    assert show_string(actual) == strip_margin(
        """
        |+-------------+-----------+-----------+-----+--------------+----------+---+---+
        ||column_number|column_name|column_type|count|count_distinct|count_null|min|max|
        |+-------------+-----------+-----------+-----+--------------+----------+---+---+
        ||            0|       FROM|    INTEGER|    1|             1|         0|  1|  1|
        ||            1|  UNION.ALL|     STRING|    1|             1|         0|  a|  a|
        |+-------------+-----------+-----------+-----+--------------+----------+---+---+
        |""",
    )


def test_analyze_with_array_struct_array(spark: SparkSession):
    """
    GIVEN a DataFrame containing an ARRAY<STRUCT<ARRAY<INT>>>
    WHEN we analyze it
    THEN no crash should occur
    """
    query = """SELECT ARRAY(STRUCT(ARRAY(1, 2, 3) as b)) as a"""
    df = spark.sql(query)
    actual = analyze(df)
    assert show_string(actual) == strip_margin(
        """
        |+-------------+-----------+-----------+-----+--------------+----------+---+---+
        ||column_number|column_name|column_type|count|count_distinct|count_null|min|max|
        |+-------------+-----------+-----------+-----+--------------+----------+---+---+
        ||            0|      a!.b!|    INTEGER|    3|             3|         0|  1|  3|
        |+-------------+-----------+-----------+-----+--------------+----------+---+---+
        |""",
    )


def test_analyze_with_bytes(spark: SparkSession):
    """
    GIVEN a DataFrame containing a column of type bytes
    WHEN we analyze it
    THEN no crash should occur
    """
    query = r"""SELECT cast('/+A=' as BINARY) as s"""
    df = spark.sql(query)
    actual = analyze(df)
    assert show_string(actual) == strip_margin(
        """
        |+-------------+-----------+-----------+-----+--------------+----------+----+----+
        ||column_number|column_name|column_type|count|count_distinct|count_null| min| max|
        |+-------------+-----------+-----------+-----+--------------+----------+----+----+
        ||            0|          s|     BINARY|    1|             1|         0|/+A=|/+A=|
        |+-------------+-----------+-----------+-----+--------------+----------+----+----+
        |""",
    )


def test_analyze_with_nested_field_in_group_and_array_column(spark: SparkSession):
    """
    GIVEN a DataFrame containing a STRUCT and an array column
    WHEN we analyze it by grouping on a column inside this struct
    THEN no crash should occur
    """
    query = """SELECT 1 as id, STRUCT(2 as b, 3 as c) as a, ARRAY(1, 2, 3) as arr"""
    df = spark.sql(query)
    actual = analyze(df, group_by="a.b")
    assert show_string(actual) == strip_margin(
        """
        |+-----+-------------+-----------+-----------+-----+--------------+----------+---+---+
        ||group|column_number|column_name|column_type|count|count_distinct|count_null|min|max|
        |+-----+-------------+-----------+-----------+-----+--------------+----------+---+---+
        ||  {2}|            0|         id|    INTEGER|    1|             1|         0|  1|  1|
        ||  {2}|            2|        a.c|    INTEGER|    1|             1|         0|  3|  3|
        ||  {2}|            3|       arr!|    INTEGER|    3|             3|         0|  1|  3|
        |+-----+-------------+-----------+-----------+-----+--------------+----------+---+---+
        |""",
    )


def test_analyze_with_struct_in_group_and_array_column(spark: SparkSession):
    """
    GIVEN a DataFrame containing a STRUCT and an array column
    WHEN we analyze it by grouping on the struct column
    THEN no crash should occur
    """
    query = """SELECT 1 as id, STRUCT(2 as b, 3 as c) as a, ARRAY(1, 2, 3) as arr"""
    df = spark.sql(query)
    actual = analyze(df, group_by="a")
    assert show_string(actual) == strip_margin(
        """
        |+--------+-------------+-----------+-----------+-----+--------------+----------+---+---+
        ||   group|column_number|column_name|column_type|count|count_distinct|count_null|min|max|
        |+--------+-------------+-----------+-----------+-----+--------------+----------+---+---+
        ||{{2, 3}}|            0|         id|    INTEGER|    1|             1|         0|  1|  1|
        ||{{2, 3}}|            3|       arr!|    INTEGER|    3|             3|         0|  1|  3|
        |+--------+-------------+-----------+-----------+-----+--------------+----------+---+---+
        |""",
    )
