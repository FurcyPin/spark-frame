from pyspark.sql import Column, SparkSession
from pyspark.sql.types import DataType, IntegerType

from spark_frame import nested
from spark_frame.transformations import transform_all_fields
from spark_frame.utils import schema_string, show_string, strip_margin

WEIRD_CHARS = "_!:;,?./§*ù%µ$£&é" "(-è_çà)=#{[|^@]}"


def test_transform_all_fields_with_weird_column_names(spark: SparkSession):
    df = spark.sql(
        f"""SELECT
        "John" as `name`,
        ARRAY(STRUCT(1 as `a{WEIRD_CHARS}`), STRUCT(2 `a{WEIRD_CHARS}`)) as s1,
        ARRAY(ARRAY(1, 2), ARRAY(3, 4)) as s2,
        ARRAY(ARRAY(STRUCT(1 `a{WEIRD_CHARS}`)), ARRAY(STRUCT(2 `a{WEIRD_CHARS}`))) as s3,
        ARRAY(STRUCT(ARRAY(1, 2) `a{WEIRD_CHARS}`), STRUCT(ARRAY(3, 4) `a{WEIRD_CHARS}`)) as s4,
        ARRAY(
            STRUCT(ARRAY(
                STRUCT(STRUCT(1 as `c{WEIRD_CHARS}`) as `b{WEIRD_CHARS}`),
                STRUCT(STRUCT(2 as `c{WEIRD_CHARS}`) as `b{WEIRD_CHARS}`)
            ) `a{WEIRD_CHARS}`),
            STRUCT(ARRAY(
                STRUCT(STRUCT(3 as `c{WEIRD_CHARS}`) as `b{WEIRD_CHARS}`),
                STRUCT(STRUCT(4 as `c{WEIRD_CHARS}`) as `b{WEIRD_CHARS}`)
            ) `a{WEIRD_CHARS}`)
        ) as s5
    """
    )
    assert nested.schema_string(df) == strip_margin(
        f"""
        |root
        | |-- name: string (nullable = false)
        | |-- s1!.a{WEIRD_CHARS}: integer (nullable = false)
        | |-- s2!!: integer (nullable = false)
        | |-- s3!!.a{WEIRD_CHARS}: integer (nullable = false)
        | |-- s4!.a{WEIRD_CHARS}!: integer (nullable = false)
        | |-- s5!.a{WEIRD_CHARS}!.b{WEIRD_CHARS}.c{WEIRD_CHARS}: integer (nullable = false)
        |"""
    )
    assert show_string(df, truncate=False) == strip_margin(
        """
        |+----+----------+----------------+--------------+--------------------+------------------------------------+
        ||name|s1        |s2              |s3            |s4                  |s5                                  |
        |+----+----------+----------------+--------------+--------------------+------------------------------------+
        ||John|[{1}, {2}]|[[1, 2], [3, 4]]|[[{1}], [{2}]]|[{[1, 2]}, {[3, 4]}]|[{[{{1}}, {{2}}]}, {[{{3}}, {{4}}]}]|
        |+----+----------+----------------+--------------+--------------------+------------------------------------+
        |"""
    )

    def cast_int_as_double(col: Column, data_type: DataType):
        if isinstance(data_type, IntegerType):
            return col.cast("DOUBLE")

    actual = transform_all_fields(df, cast_int_as_double)
    assert nested.schema_string(actual) == strip_margin(
        f"""
        |root
        | |-- name: string (nullable = false)
        | |-- s1!.a{WEIRD_CHARS}: double (nullable = false)
        | |-- s2!!: double (nullable = false)
        | |-- s3!!.a{WEIRD_CHARS}: double (nullable = false)
        | |-- s4!.a{WEIRD_CHARS}!: double (nullable = false)
        | |-- s5!.a{WEIRD_CHARS}!.b{WEIRD_CHARS}.c{WEIRD_CHARS}: double (nullable = false)
        |"""
    )
    assert show_string(actual.select("name", "s1", "s2", "s3"), truncate=False) == strip_margin(
        """
        |+----+--------------+------------------------+------------------+
        ||name|s1            |s2                      |s3                |
        |+----+--------------+------------------------+------------------+
        ||John|[{1.0}, {2.0}]|[[1.0, 2.0], [3.0, 4.0]]|[[{1.0}], [{2.0}]]|
        |+----+--------------+------------------------+------------------+
        |"""
    )
    assert show_string(actual.select("name", "s4", "s5"), truncate=False) == strip_margin(
        """
        |+----+----------------------------+--------------------------------------------+
        ||name|s4                          |s5                                          |
        |+----+----------------------------+--------------------------------------------+
        ||John|[{[1.0, 2.0]}, {[3.0, 4.0]}]|[{[{{1.0}}, {{2.0}}]}, {[{{3.0}}, {{4.0}}]}]|
        |+----+----------------------------+--------------------------------------------+
        |"""
    )


def test_transform_all_fields_with_maps_and_weird_column_names(spark: SparkSession):
    df = spark.sql(
        f"""SELECT
        "John" as `name{WEIRD_CHARS}`,
        MAP(1, 2) as `m1{WEIRD_CHARS}`,
        MAP(STRUCT(1 as `a{WEIRD_CHARS}`), STRUCT(2 as `b{WEIRD_CHARS}`)) as `m2{WEIRD_CHARS}`
    """
    )
    assert schema_string(df) == strip_margin(
        f"""
        |root
        | |-- name{WEIRD_CHARS}: string (nullable = false)
        | |-- m1{WEIRD_CHARS}: map (nullable = false)
        | |    |-- key: integer
        | |    |-- value: integer (valueContainsNull = false)
        | |-- m2{WEIRD_CHARS}: map (nullable = false)
        | |    |-- key: struct
        | |    |    |-- a{WEIRD_CHARS}: integer (nullable = false)
        | |    |-- value: struct (valueContainsNull = false)
        | |    |    |-- b{WEIRD_CHARS}: integer (nullable = false)
        |"""
    )
    renamed_df = (
        df.withColumnRenamed(f"name{WEIRD_CHARS}", "name")
        .withColumnRenamed(f"m1{WEIRD_CHARS}", "m1")
        .withColumnRenamed(f"m2{WEIRD_CHARS}", "m2")
    )
    assert show_string(renamed_df, truncate=False) == strip_margin(
        """
        |+----+--------+------------+
        ||name|m1      |m2          |
        |+----+--------+------------+
        ||John|{1 -> 2}|{{1} -> {2}}|
        |+----+--------+------------+
        |"""
    )

    def cast_int_as_double(col: Column, data_type: DataType):
        if isinstance(data_type, IntegerType):
            return col.cast("DOUBLE")

    actual = transform_all_fields(df, cast_int_as_double)
    renamed_actual = (
        actual.withColumnRenamed(f"name{WEIRD_CHARS}", "name")
        .withColumnRenamed(f"m1{WEIRD_CHARS}", "m1")
        .withColumnRenamed(f"m2{WEIRD_CHARS}", "m2")
    )
    renamed_actual.show(truncate=False)
    assert schema_string(actual) == strip_margin(
        f"""
        |root
        | |-- name{WEIRD_CHARS}: string (nullable = false)
        | |-- m1{WEIRD_CHARS}: map (nullable = false)
        | |    |-- key: double
        | |    |-- value: double (valueContainsNull = false)
        | |-- m2{WEIRD_CHARS}: map (nullable = false)
        | |    |-- key: struct
        | |    |    |-- a{WEIRD_CHARS}: double (nullable = false)
        | |    |-- value: struct (valueContainsNull = false)
        | |    |    |-- b{WEIRD_CHARS}: double (nullable = false)
        |"""
    )
    assert show_string(renamed_actual, truncate=False) == strip_margin(
        """
        |+----+------------+----------------+
        ||name|m1          |m2              |
        |+----+------------+----------------+
        ||John|{1.0 -> 2.0}|{{1.0} -> {2.0}}|
        |+----+------------+----------------+
        |"""
    )
