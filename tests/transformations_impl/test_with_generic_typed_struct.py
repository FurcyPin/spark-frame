from decimal import Decimal

from pyspark.sql import Row, SparkSession

from spark_frame import transformations

generic_typed_struct_type = """
ARRAY<STRUCT<
    key:STRING,
    type:STRING,
    value:STRUCT<
        boolean:BOOLEAN,
        bytes:BINARY,
        date:DATE,
        float:DOUBLE,
        int:BIGINT,
        string:STRING,
        timestamp:TIMESTAMP
    >
>>
"""


def test_with_generic_typed_struct(spark: SparkSession):
    df = spark.createDataFrame(
        [
            (1, {"name": "Paris"}, {"first_name": "Jacques", "last_name": "Dupont", "is_adult": True}),
            (2, {"name": "Paris"}, {"first_name": "Michel", "last_name": "Roger", "is_adult": False}),
            (3, {"name": "Paris"}, {"first_name": "Marie", "last_name": "De La Rue", "is_adult": True}),
        ],
        "id INT, location STRUCT<name:STRING>, " "person STRUCT<first_name:STRING, last_name:STRING, is_adult:BOOLEAN>",
    )
    expected_df = spark.createDataFrame(
        [
            Row(
                id=1,
                location=[
                    Row(
                        key="name",
                        type="string",
                        value=Row(
                            boolean=None,
                            bytes=None,
                            date=None,
                            float=None,
                            int=None,
                            string="Paris",
                            timestamp=None,
                        ),
                    )
                ],
                person=[
                    Row(
                        key="first_name",
                        type="string",
                        value=Row(
                            boolean=None,
                            bytes=None,
                            date=None,
                            float=None,
                            int=None,
                            string="Jacques",
                            timestamp=None,
                        ),
                    ),
                    Row(
                        key="last_name",
                        type="string",
                        value=Row(
                            boolean=None,
                            bytes=None,
                            date=None,
                            float=None,
                            int=None,
                            string="Dupont",
                            timestamp=None,
                        ),
                    ),
                    Row(
                        key="is_adult",
                        type="boolean",
                        value=Row(
                            boolean=True, bytes=None, date=None, float=None, int=None, string=None, timestamp=None
                        ),
                    ),
                ],
            ),
            Row(
                id=2,
                location=[
                    Row(
                        key="name",
                        type="string",
                        value=Row(
                            boolean=None,
                            bytes=None,
                            date=None,
                            float=None,
                            int=None,
                            string="Paris",
                            timestamp=None,
                        ),
                    )
                ],
                person=[
                    Row(
                        key="first_name",
                        type="string",
                        value=Row(
                            boolean=None,
                            bytes=None,
                            date=None,
                            float=None,
                            int=None,
                            string="Michel",
                            timestamp=None,
                        ),
                    ),
                    Row(
                        key="last_name",
                        type="string",
                        value=Row(
                            boolean=None,
                            bytes=None,
                            date=None,
                            float=None,
                            int=None,
                            string="Roger",
                            timestamp=None,
                        ),
                    ),
                    Row(
                        key="is_adult",
                        type="boolean",
                        value=Row(
                            boolean=False, bytes=None, date=None, float=None, int=None, string=None, timestamp=None
                        ),
                    ),
                ],
            ),
            Row(
                id=3,
                location=[
                    Row(
                        key="name",
                        type="string",
                        value=Row(
                            boolean=None,
                            bytes=None,
                            date=None,
                            float=None,
                            int=None,
                            string="Paris",
                            timestamp=None,
                        ),
                    )
                ],
                person=[
                    Row(
                        key="first_name",
                        type="string",
                        value=Row(
                            boolean=None,
                            bytes=None,
                            date=None,
                            float=None,
                            int=None,
                            string="Marie",
                            timestamp=None,
                        ),
                    ),
                    Row(
                        key="last_name",
                        type="string",
                        value=Row(
                            boolean=None,
                            bytes=None,
                            date=None,
                            float=None,
                            int=None,
                            string="De La Rue",
                            timestamp=None,
                        ),
                    ),
                    Row(
                        key="is_adult",
                        type="boolean",
                        value=Row(
                            boolean=True, bytes=None, date=None, float=None, int=None, string=None, timestamp=None
                        ),
                    ),
                ],
            ),
        ],
        f"""id INT, location {generic_typed_struct_type}, person {generic_typed_struct_type}""",
    )
    result = transformations.with_generic_typed_struct(df, ["location", "person"])
    assert result.sort("id").collect() == expected_df.sort("id").collect()


def test_with_generic_typed_struct_with_decimal_types(spark: SparkSession):
    df = spark.createDataFrame(
        [
            (1, {"a": "A", "decimal_col": Decimal(1.1)}),
            (2, {"a": "B", "decimal_col": Decimal(2.2)}),
            (3, {"a": "C", "decimal_col": Decimal(3.3)}),
        ],
        "id INT, s STRUCT<a:STRING, decimal_col:DECIMAL(10, 2)>",
    )
    expected_df = spark.createDataFrame(
        [
            Row(
                id=1,
                s=[
                    Row(
                        key="a",
                        type="string",
                        value=Row(
                            date=None, timestamp=None, int=None, float=None, boolean=None, string="A", bytes=None
                        ),
                    ),
                    Row(
                        key="decimal_col",
                        type="float",
                        value=Row(
                            date=None, timestamp=None, int=None, float=1.1, boolean=None, string=None, bytes=None
                        ),
                    ),
                ],
            ),
            Row(
                id=2,
                s=[
                    Row(
                        key="a",
                        type="string",
                        value=Row(
                            date=None, timestamp=None, int=None, float=None, boolean=None, string="B", bytes=None
                        ),
                    ),
                    Row(
                        key="decimal_col",
                        type="float",
                        value=Row(
                            date=None, timestamp=None, int=None, float=2.2, boolean=None, string=None, bytes=None
                        ),
                    ),
                ],
            ),
            Row(
                id=3,
                s=[
                    Row(
                        key="a",
                        type="string",
                        value=Row(
                            date=None, timestamp=None, int=None, float=None, boolean=None, string="C", bytes=None
                        ),
                    ),
                    Row(
                        key="decimal_col",
                        type="float",
                        value=Row(
                            date=None, timestamp=None, int=None, float=3.3, boolean=None, string=None, bytes=None
                        ),
                    ),
                ],
            ),
        ],
        f"""id INT, s {generic_typed_struct_type}""",
    ).withColumnRenamed("person", "person.struct")
    result = transformations.with_generic_typed_struct(df, ["s"])
    assert result.sort("id").collect() == expected_df.sort("id").collect()


def test_with_generic_typed_struct_with_unsupported_types(spark: SparkSession):
    df = spark.createDataFrame(
        [
            (1, {"a": "A", "array_col": [1, 2, 3]}),
            (2, {"a": "B", "array_col": [1, 2, 3]}),
            (3, {"a": "C", "array_col": [1, 2, 3]}),
        ],
        "id INT, s STRUCT<a:STRING, array_col:ARRAY<INT>>",
    )
    expected_df = spark.createDataFrame(
        [
            Row(
                id=1,
                s=[
                    Row(
                        key="a",
                        type="string",
                        value=Row(
                            date=None, timestamp=None, int=None, float=None, boolean=None, string="A", bytes=None
                        ),
                    ),
                ],
            ),
            Row(
                id=2,
                s=[
                    Row(
                        key="a",
                        type="string",
                        value=Row(
                            date=None, timestamp=None, int=None, float=None, boolean=None, string="B", bytes=None
                        ),
                    ),
                ],
            ),
            Row(
                id=3,
                s=[
                    Row(
                        key="a",
                        type="string",
                        value=Row(
                            date=None, timestamp=None, int=None, float=None, boolean=None, string="C", bytes=None
                        ),
                    ),
                ],
            ),
        ],
        f"""id INT, s {generic_typed_struct_type}""",
    ).withColumnRenamed("person", "person.struct")
    result = transformations.with_generic_typed_struct(df, ["s"])
    assert result.sort("id").collect() == expected_df.sort("id").collect()


def test_with_generic_typed_struct_with_weird_column_names(spark: SparkSession):
    df = spark.createDataFrame(
        [
            (1, {"c.`.d": "A"}),
            (2, {"c.`.d": "B"}),
            (3, {"c.`.d": "C"}),
        ],
        "id INT, `a.``.b` STRUCT<`c.``.d`:STRING>",
    )
    expected_df = spark.createDataFrame(
        [
            Row(
                id=1,
                s=[
                    Row(
                        key="c.`.d",
                        type="string",
                        value=Row(
                            date=None, timestamp=None, int=None, float=None, boolean=None, string="A", bytes=None
                        ),
                    ),
                ],
            ),
            Row(
                id=2,
                s=[
                    Row(
                        key="c.`.d",
                        type="string",
                        value=Row(
                            date=None, timestamp=None, int=None, float=None, boolean=None, string="B", bytes=None
                        ),
                    ),
                ],
            ),
            Row(
                id=3,
                s=[
                    Row(
                        key="c.`.d",
                        type="string",
                        value=Row(
                            date=None, timestamp=None, int=None, float=None, boolean=None, string="C", bytes=None
                        ),
                    ),
                ],
            ),
        ],
        f"""id INT, s {generic_typed_struct_type}""",
    ).withColumnRenamed("s", "a.`.b")
    result = transformations.with_generic_typed_struct(df, ["`a.``.b`"])
    assert result.sort("id").collect() == expected_df.sort("id").collect()
