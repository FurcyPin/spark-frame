from pyspark.sql import SparkSession

from spark_frame import transformations


def test_parse_json_columns(spark: SparkSession):
    """
    GIVEN a DataFrame with two json columns
    WHEN using parse_json_columns
    THEN the resulting columns should be properly parsed
    """
    df = spark.createDataFrame(
        [
            (1, '{"a": 1, "b": 1}', '{"c": 1, "d": 1}'),
            (2, '{"a": 2, "b": 2}', '{"c": 2, "d": 2}'),
            (3, '{"a": 3, "b": 3}', '{"c": 3, "d": 3}'),
        ],
        "id INT, json1 STRING, json2 STRING",
    )
    df2 = transformations.parse_json_columns(df, ["json1", "json2"])

    json1_type = [col[1] for col in df2.dtypes if col[0] == "json1"]

    assert json1_type[0], "struct<a:bigint,b:bigint>"

    expected = spark.createDataFrame(
        [
            (1, {"a": 1, "b": 1}, {"c": 1, "d": 1}),
            (2, {"a": 2, "b": 2}, {"c": 2, "d": 2}),
            (3, {"a": 3, "b": 3}, {"c": 3, "d": 3}),
        ],
        "id INT, json1 STRUCT<a: BIGINT, b: BIGINT>, json2 STRUCT<c: BIGINT, d:BIGINT>",
    )

    assert df2.sort("id").collect() == expected.sort("id").collect()


def test_parse_json_columns_with_arrays(spark: SparkSession):
    """
    GIVEN a DataFrame with two json columns representing arrays
    WHEN using parse_json_columns
    THEN the resulting columns should be properly parsed
    """
    df = spark.createDataFrame(
        [
            (1, '[{"a": 1}, {"a": 2}]', '[{"c": 1}, {"c": 2}]'),
            (2, '[{"a": 2}, {"a": 4}]', '[{"c": 2}, {"c": 4}]'),
            (3, '[{"a": 3}, {"a": 6}]', '[{"c": 3}, {"c": 6}]'),
        ],
        "id INT, json1 STRING, json2 STRING",
    )
    df2 = transformations.parse_json_columns(df, ["json1", "json2"])

    json1_type = [col[1] for col in df2.dtypes if col[0] == "json1"]

    assert json1_type[0], "array<struct<a:bigint>>"

    expected = spark.createDataFrame(
        [
            (1, [{"a": 1}, {"a": 2}], [{"c": 1}, {"c": 2}]),
            (2, [{"a": 2}, {"a": 4}], [{"c": 2}, {"c": 4}]),
            (3, [{"a": 3}, {"a": 6}], [{"c": 3}, {"c": 6}]),
        ],
        "id INT, json1 ARRAY<STRUCT<a: BIGINT>>, json2 ARRAY<STRUCT<c: BIGINT>>",
    )

    assert df2.sort("id").collect() == expected.sort("id").collect()


def test_parse_json_columns_with_empty_array(spark: SparkSession):
    """
    GIVEN a DataFrame with a json column representing an empty array
    WHEN using parse_json_columns
    THEN the resulting columns should be properly parsed
    """
    # fmt: off
    df = spark.createDataFrame(
        [(1, '[{"a": 1}, {"a": 2}]'),
         (2, "[]"),
         (3, '[{"a": 3}, {"a": 6}]')],
        "id INT, json1 STRING"
    )
    df2 = transformations.parse_json_columns(df, ["json1"])

    expected = spark.createDataFrame(
        [(1, [{"a": 1}, {"a": 2}]),
         (2, []),
         (3, [{"a": 3}, {"a": 6}])],
        "id INT, json1 ARRAY<STRUCT<a: BIGINT>>"
    )
    # fmt: on

    assert df2.sort("id").collect() == expected.sort("id").collect()


def test_parse_json_columns_with_nulls(spark: SparkSession):
    """
    GIVEN a DataFrame with a json column that is sometimes NULL
    WHEN using parse_json_columns
    THEN the resulting columns should be properly parsed
    """
    # fmt: off
    df = spark.createDataFrame(
        [(1, '[{"a": 1}, {"a": 2}]'),
         (2, None),
         (3, '[{"a": 3}, {"a": 6}]')],
        "id INT, json1 STRING"
    )
    df2 = transformations.parse_json_columns(df, ["json1"])

    expected = spark.createDataFrame(
        [(1, [{"a": 1}, {"a": 2}]),
         (2, None),
         (3, [{"a": 3}, {"a": 6}])],
        "id INT, json1 ARRAY<STRUCT<a: BIGINT>>"
    )
    # fmt: on

    assert df2.sort("id").collect() == expected.sort("id").collect()
