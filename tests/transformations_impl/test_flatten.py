from pyspark.sql import SparkSession

from spark_frame import transformations


def test_flatten(spark: SparkSession):
    # fmt: off
    df = spark.createDataFrame(
        [(1, {"a": 1, "b": {"c": 2, "d": 3}}),
         (2, None),
         (3, {"a": 1, "b": {"c": 2, "d": 3}})],
        "id INT, s STRUCT<a:INT, b:STRUCT<c:INT, d:INT>>",
    )
    # fmt: on
    df2 = transformations.flatten(df)

    # fmt: off
    expected = spark.createDataFrame(
        [(1, 1, 2, 3),
         (2, None, None, None),
         (3, 1, 2, 3)],
        "id INT, `s.a` INT, `s.b.c` INT, `s.b.d` INT",
    )
    # fmt: on
    assert df2.sort("id").collect() == expected.sort("id").collect()


def test_flatten_with_complex_names(spark: SparkSession):
    df = spark.createDataFrame(
        [
            (1, {"a.a1": 1, "b.b1": {"c.c1": 2, "d.d1": 3}}),
            (2, None),
            (3, {"a.a1": 1, "b.b1": {"c.c1": 2, "d.d1": 3}}),
        ],
        "`id.id1` INT, `s.s1` STRUCT<`a.a1`:INT, `b.b1`:STRUCT<`c.c1`:INT, `d.d1`:INT>>",
    )
    df2 = transformations.flatten(df, "?")

    # fmt: off
    expected = spark.createDataFrame(
        [(1, 1, 2, 3),
         (2, None, None, None),
         (3, 1, 2, 3)],
        "`id.id1` INT, `s.s1?a.a1` INT, `s.s1?b.b1?c.c1` INT, `s.s1?b.b1?d.d1` INT",
    )
    # fmt: on
    assert df2.sort("`id.id1`").collect() == expected.sort("`id.id1`").collect()
