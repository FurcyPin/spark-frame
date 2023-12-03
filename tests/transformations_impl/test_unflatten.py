from pyspark.sql import SparkSession

from spark_frame import transformations


def test_unflatten(spark: SparkSession):
    # fmt: off
    df = spark.createDataFrame(
        [(1, 1, 2, 3),
         (2, None, None, None),
         (3, 1, 2, 3)],
        "id INT, `s.a` INT, `s.b.c` INT, `s.b.d` INT",
    )

    df2 = transformations.unflatten(df)

    expected = spark.createDataFrame(
        [(1, {"a": 1, "b": {"c": 2, "d": 3}}),
         (2, None),
         (3, {"a": 1, "b": {"c": 2, "d": 3}})],
        "id INT, s STRUCT<a:INT, b:STRUCT<c:INT, d:INT>>",
    )
    # fmt: on
    assert df2.sort("id").collect() == expected.sort("id").collect()


def test_unflatten_with_complex_names(spark: SparkSession):
    # fmt: off
    df = spark.createDataFrame(
        [(1, 1, 2, 3),
         (2, None, None, None),
         (3, 1, 2, 3)],
        "id INT, `s.s1?a.a1` INT, `s.s1?b.b1?c.c1` INT, `s.s1?b.b1?d.d1` INT",
    )
    # fmt: on

    df2 = transformations.unflatten(df, "?")

    expected = spark.createDataFrame(
        [
            (1, {"a.a1": 1, "b.b1": {"c.c1": 2, "d.d1": 3}}),
            (2, None),
            (3, {"a.a1": 1, "b.b1": {"c.c1": 2, "d.d1": 3}}),
        ],
        "id INT, `s.s1` STRUCT<`a.a1`:INT, `b.b1`:STRUCT<`c.c1`:INT, `d.d1`:INT>>",
    )
    assert df2.sort("id").collect() == expected.sort("id").collect()


def test_unflatten_with_struct_and_col(spark: SparkSession):
    """When we have a s.s1 struct and a `s.s2` column"""
    df = spark.createDataFrame([({"s1": 1}, 2)], "s STRUCT<s1: INT>, `s.s2` INT")

    df2 = transformations.unflatten(df)

    expected = spark.createDataFrame([({"s1": 1, "s2": 2},)], "s STRUCT<s1: INT, s2: INT>")

    assert df2.collect() == expected.collect()


def test_unflatten_with_struct_and_col_2(spark: SparkSession):
    """When we have a r.s.s1 struct and a r.`s.s2` column"""
    df = spark.createDataFrame([({"s": {"s1": 1}, "s.s2": 2},)], "r STRUCT<s: STRUCT<s1: INT>, `s.s2`: INT>")

    df2 = transformations.unflatten(df)

    expected = spark.createDataFrame([({"s": {"s1": 1, "s2": 2}},)], "r STRUCT<s: STRUCT<s1: INT, s2: INT>>")

    assert df2.collect() == expected.collect()
