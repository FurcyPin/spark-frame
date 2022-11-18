import pytest
from pyspark.sql import SparkSession


@pytest.fixture(autouse=True)
def spark() -> SparkSession:
    from pyspark.sql import SparkSession

    return SparkSession.builder.appName("doctest").getOrCreate()
