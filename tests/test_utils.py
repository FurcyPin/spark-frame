import pytest
from pyspark.sql import SparkSession

from spark_frame import utils


def test_get_nested_col_type_from_schema_with_wrong_column_name(spark: SparkSession):
    df = spark.createDataFrame(
        [
            (1, {"a": "A"}),
            (2, {"a": "B"}),
            (3, {"a": "C"}),
        ],
        "id INT, s STRUCT<a:STRING>",
    )
    with pytest.raises(ValueError) as e:
        utils.get_nested_col_type_from_schema("s.b", df.schema)
    assert str(e.value) == 'Cannot resolve column name "s.b"'
