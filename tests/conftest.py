import pytest
from pyspark.sql import SparkSession

from conftest import fix_pyspark_show_change

fix_pyspark_show_change = fix_pyspark_show_change


@pytest.fixture(autouse=True)
def spark() -> SparkSession:
    from pyspark.sql import SparkSession

    return SparkSession.builder.appName("doctest").getOrCreate()
