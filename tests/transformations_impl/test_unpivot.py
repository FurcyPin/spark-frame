from pyspark.sql import SparkSession

from spark_frame import transformations


def test_unpivot_with_complex_col_names(spark: SparkSession):
    spark = spark
    df = spark.createDataFrame(
        [
            (2018, "Orange", None, 4000, None),
            (2018, "Beans", None, 1500, 2000),
            (2018, "Banana", 2000, 400, None),
            (2018, "Carrots", 2000, 1200, None),
            (2019, "Orange", 5000, None, 5000),
            (2019, "Beans", None, 1500, 2000),
            (2019, "Banana", None, 1400, 400),
            (2019, "Carrots", None, 200, None),
        ],
        "year INT, `product.type` STRING, `country.Canada` INT, `country.China` INT, `country.Mexico` INT",
    )

    actual = transformations.unpivot(df, ["year", "product.type"], key_alias="country.name", value_alias="total")
    expected = spark.createDataFrame(
        [
            (2018, "Banana", "country.Canada", 2000),
            (2018, "Banana", "country.China", 400),
            (2018, "Banana", "country.Mexico", None),
            (2018, "Beans", "country.Canada", None),
            (2018, "Beans", "country.China", 1500),
            (2018, "Beans", "country.Mexico", 2000),
            (2018, "Carrots", "country.Canada", 2000),
            (2018, "Carrots", "country.China", 1200),
            (2018, "Carrots", "country.Mexico", None),
            (2018, "Orange", "country.Canada", None),
            (2018, "Orange", "country.China", 4000),
            (2018, "Orange", "country.Mexico", None),
            (2019, "Banana", "country.Canada", None),
            (2019, "Banana", "country.China", 1400),
            (2019, "Banana", "country.Mexico", 400),
            (2019, "Beans", "country.Canada", None),
            (2019, "Beans", "country.China", 1500),
            (2019, "Beans", "country.Mexico", 2000),
            (2019, "Carrots", "country.Canada", None),
            (2019, "Carrots", "country.China", 200),
            (2019, "Carrots", "country.Mexico", None),
            (2019, "Orange", "country.Canada", 5000),
            (2019, "Orange", "country.China", None),
            (2019, "Orange", "country.Mexico", 5000),
        ],
        "year INT, `product.type` STRING, `country.name` STRING, total INT",
    )
    actual_sorted = actual.sort("year", "`product.type`", "`country.name`")
    expected_sorted = expected.sort("year", "`product.type`", "`country.name`")
    assert actual_sorted.collect() == expected_sorted.collect()
