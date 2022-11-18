import re
from typing import List, Optional, cast

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import StructField, StructType

from spark_frame.utils import (
    assert_true,
    get_nested_col_type_from_schema,
    quote,
    unquote,
)


def with_generic_typed_struct(df: DataFrame, col_names: List[str]) -> DataFrame:
    """Transform the specified struct columns of a given [Dataframe][pyspark.sql.DataFrame] into
    generic typed struct columns with the following generic schema
    (based on [https://spark.apache.org/docs/latest/sql-ref-datatypes.html](
    https://spark.apache.org/docs/latest/sql-ref-datatypes.html)) :

        STRUCT<
            key: STRING, -- (name of the field inside the struct)
            type: STRING, -- (type of the field inside the struct)
            value: STRUCT< -- (all the fields will be null except for the one with the correct type)
                date: DATE,
                timestamp: TIMESTAMP,
                int: LONG,
                float: DOUBLE,
                boolean: BOOLEAN,
                string: STRING,
                bytes: BINARY
            >
        >

    The following spark types will be automatically cast into the more generic following types:

    - `tinyint`, `smallint`, `int` -> `bigint`
    - `float`, `decimal` -> `double`

    Args:
        df: The Dataframe to transform
        col_names: A list of column names to transform

    Returns:
        A Dataframe with the columns transformed into generic typed structs

    !!! warning "Limitations"
        Currently, complex field types (structs, maps, arrays) are not supported.
        All fields of the struct columns to convert must be of basic types.

    Examples:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.createDataFrame(
        ...     [(1, {"first.name": "Jacques", "age": 25, "is.an.adult": True}),
        ...      (2, {"first.name": "Michel", "age": 12, "is.an.adult": False}),
        ...      (3, {"first.name": "Marie", "age": 36, "is.an.adult": True})],
        ...     "id INT, `person.struct` STRUCT<`first.name`:STRING, age:INT, `is.an.adult`:BOOLEAN>"
        ... )
        >>> df.show(truncate=False)
        +---+-------------------+
        |id |person.struct      |
        +---+-------------------+
        |1  |{Jacques, 25, true}|
        |2  |{Michel, 12, false}|
        |3  |{Marie, 36, true}  |
        +---+-------------------+
        <BLANKLINE>
        >>> res = with_generic_typed_struct(df, ["`person.struct`"])
        >>> res.printSchema()
        root
         |-- id: integer (nullable = true)
         |-- person.struct: array (nullable = false)
         |    |-- element: struct (containsNull = false)
         |    |    |-- key: string (nullable = false)
         |    |    |-- type: string (nullable = false)
         |    |    |-- value: struct (nullable = false)
         |    |    |    |-- boolean: boolean (nullable = true)
         |    |    |    |-- bytes: binary (nullable = true)
         |    |    |    |-- date: date (nullable = true)
         |    |    |    |-- float: double (nullable = true)
         |    |    |    |-- int: long (nullable = true)
         |    |    |    |-- string: string (nullable = true)
         |    |    |    |-- timestamp: timestamp (nullable = true)
        <BLANKLINE>
        >>> res.show(10, False) # noqa: E501
        +---+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |id |person.struct                                                                                                                                                                                  |
        +---+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |1  |[{first.name, string, {null, null, null, null, null, Jacques, null}}, {age, int, {null, null, null, null, 25, null, null}}, {is.an.adult, boolean, {true, null, null, null, null, null, null}}]|
        |2  |[{first.name, string, {null, null, null, null, null, Michel, null}}, {age, int, {null, null, null, null, 12, null, null}}, {is.an.adult, boolean, {false, null, null, null, null, null, null}}]|
        |3  |[{first.name, string, {null, null, null, null, null, Marie, null}}, {age, int, {null, null, null, null, 36, null, null}}, {is.an.adult, boolean, {true, null, null, null, null, null, null}}]  |
        +---+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        <BLANKLINE>
    """

    source_to_cast = {
        "date": "date",
        "timestamp": "timestamp",
        "tinyint": "bigint",
        "smallint": "bigint",
        "int": "bigint",
        "bigint": "bigint",
        "float": "double",
        "double": "double",
        "boolean": "boolean",
        "string": "string",
        "binary": "binary",
    }
    """Mapping indicating for each source Spark DataTypes the type into which it will be cast."""

    cast_to_name = {
        "binary": "bytes",
        "bigint": "int",
        "double": "float",
    }
    """Mapping indicating for each already cast Spark DataTypes the name of the corresponding field.
    When missing, the same name will be kept."""

    name_cast = {cast_to_name.get(value, value): value for value in source_to_cast.values()}
    # We make sure the types are sorted
    name_cast = {k: v for k, v in sorted(name_cast.items())}

    def match_regex_types(source_type: str) -> Optional[str]:
        """Matches the source types against regexes to identify more complex types (like Decimal(x, y))"""
        regex_to_cast_types = [(re.compile("decimal(.*)"), "float")]
        for regex, cast_type in regex_to_cast_types:
            if regex.match(source_type) is not None:
                return cast_type
        return None

    def field_to_col(field: StructField, column_name: str) -> Optional[Column]:
        """Transforms the specified field into a generic column"""
        source_type = field.dataType.simpleString()
        cast_type = source_to_cast.get(source_type)
        field_name = column_name + "." + quote(field.name)
        if cast_type is None:
            cast_type = match_regex_types(source_type)
        if cast_type is None:
            print(
                "WARNING: The field {field_name} is of type {source_type} which is currently unsupported. "
                "This field will be dropped.".format(field_name=field_name, source_type=source_type)
            )
            return None
        name_type = cast_to_name.get(cast_type, cast_type)
        return f.struct(
            f.lit(field.name).alias("key"),
            f.lit(name_type).alias("type"),
            # In the code below, we use f.expr instead of f.col because it looks like f.col
            # does not support column names with backquotes in them, but f.expr does :-p
            f.struct(
                *[
                    (f.expr(field_name) if name_type == name_t else f.lit(None)).astype(cast_t).alias(name_t)
                    for name_t, cast_t in name_cast.items()
                ]
            ).alias("value"),
        )

    for col_name in col_names:
        schema = get_nested_col_type_from_schema(col_name, df.schema)
        assert_true(isinstance(schema, StructType))
        schema = cast(StructType, schema)
        columns = [field_to_col(field, col_name) for field in schema.fields]
        columns_2 = [col for col in columns if col is not None]
        df = df.withColumn(unquote(col_name), f.array(*columns_2).alias("values"))
    return df
