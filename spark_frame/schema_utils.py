import json

from pyspark import SparkContext
from pyspark.sql.types import DataType, StructType, _parse_datatype_string

from spark_frame.utils import assert_true


def schema_from_json(json_string: str) -> StructType:
    """Parses the given json string representing a Spark :class:`StructType`.

    Only schema representing StructTypes can be parsed, this means that
    `schema_from_json(schema_to_json(data_type))` will crash if `data_type` is not a StructType.

    Args:
        json_string: A string representation of a DataFrame schema.

    Returns:
        A StructType object representing the DataFrame schema

    Examples:

        >>> schema_from_json('''{"fields":[
        ...     {"metadata":{},"name":"a","nullable":true,"type":"byte"},
        ...     {"metadata":{},"name":"b","nullable":true,"type":"decimal(16,8)"}
        ... ],"type":"struct"}''')
        StructType([StructField('a', ByteType(), True), StructField('b', DecimalType(16,8), True)])
        >>> schema_from_json('''{"fields":[
        ...     {"metadata":{},"name":"a","nullable":true,"type":"double"},
        ...     {"metadata":{},"name":"b","nullable":true,"type":"string"}
        ... ],"type":"struct"}''')
        StructType([StructField('a', DoubleType(), True), StructField('b', StringType(), True)])
        >>> schema_from_json('''{"fields":[
        ... {"metadata":{},"name":"a","nullable":true,"type":{
        ...     "containsNull":true,"elementType":"short","type":"array"
        ... }}
        ... ],"type":"struct"}''')
        StructType([StructField('a', ArrayType(ShortType(), True), True)])

        **Error cases:**

        >>> schema_from_json('"integer"') # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          ...
        TypeError: string indices must be integers
        >>> schema_from_json('''{"keyType":"string","type":"map",
        ... "valueContainsNull":true,"valueType":"string"}''') # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          ...
        KeyError: 'fields'

    """
    return StructType.fromJson(json.loads(json_string))


def schema_from_simple_string(schema_string: str) -> DataType:
    """Parses the given data type string to a :class:`DataType`. The data type string format equals
    [pyspark.sql.types.DataType.simpleString][], except that the top level struct type can omit
    the ``struct<>``.
    This method requires the SparkSession to have already been instantiated.

    Args:
        schema_string: A simpleString representing a DataFrame schema.

    Returns:
        A DataType object representing the DataFrame schema.

    Raises:
        AssertionError: If no SparkContext has been instantiated first.

    Examples:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> schema_from_simple_string("int ")
        IntegerType()
        >>> schema_from_simple_string("INT ")
        IntegerType()
        >>> schema_from_simple_string("a: byte, b: decimal(  16 , 8   ) ")
        StructType([StructField('a', ByteType(), True), StructField('b', DecimalType(16,8), True)])
        >>> schema_from_simple_string("a DOUBLE, b STRING")
        StructType([StructField('a', DoubleType(), True), StructField('b', StringType(), True)])
        >>> schema_from_simple_string("a: array< short>")
        StructType([StructField('a', ArrayType(ShortType(), True), True)])
        >>> schema_from_simple_string(" map<string , string > ")
        MapType(StringType(), StringType(), True)

        **Error cases:**

        >>> schema_from_simple_string("blabla") # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          ...
        pyspark.sql.utils.ParseException:...
        >>> schema_from_simple_string("a: int,") # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          ...
        pyspark.sql.utils.ParseException:...
        >>> schema_from_simple_string("array<int") # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          ...
        pyspark.sql.utils.ParseException:...
        >>> schema_from_simple_string("map<int, boolean>>") # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          ...
        pyspark.sql.utils.ParseException:...

    """
    sc = SparkContext._active_spark_context
    assert_true(sc is not None, "No SparkContext has been instantiated yet")
    return _parse_datatype_string(schema_string)


def schema_to_json(schema: DataType) -> str:
    """Convert the given datatype into a json string.

    Args:
        schema: A DataFrame schema.

    Returns:
        A single-line json string representing the DataFrame schema.

    Examples:

        >>> from pyspark.sql.types import *
        >>> schema_to_json(IntegerType())
        '"integer"'
        >>> schema_to_json(StructType([StructField('a', ByteType(), True), StructField('b', DecimalType(16,8), True)]))
        '{"fields":[{"metadata":{},"name":"a","nullable":true,"type":"byte"},{"metadata":{},"name":"b","nullable":true,"type":"decimal(16,8)"}],"type":"struct"}'
        >>> schema_to_json(StructType([StructField('a', DoubleType(), True), StructField('b', StringType(), True)]))
        '{"fields":[{"metadata":{},"name":"a","nullable":true,"type":"double"},{"metadata":{},"name":"b","nullable":true,"type":"string"}],"type":"struct"}'
        >>> schema_to_json(StructType([StructField('a', ArrayType(ShortType(), True), True)]))
        '{"fields":[{"metadata":{},"name":"a","nullable":true,"type":{"containsNull":true,"elementType":"short","type":"array"}}],"type":"struct"}'
        >>> schema_to_json(MapType(StringType(), StringType(), True))
        '{"keyType":"string","type":"map","valueContainsNull":true,"valueType":"string"}'

    """
    return schema.json()


def schema_to_pretty_json(schema: DataType) -> str:
    """Convert the given datatype into a pretty (indented) json string.

    Args:
        schema: A DataFrame schema.

    Returns:
        A multi-line indented json string representing the DataFrame schema.

    Examples:

        >>> from pyspark.sql.types import *
        >>> print(schema_to_pretty_json(IntegerType()))
        "integer"
        >>> print(schema_to_pretty_json(StructType([StructField('a', ArrayType(ShortType(), True), True)])))
        {
          "fields": [
            {
              "metadata": {},
              "name": "a",
              "nullable": true,
              "type": {
                "containsNull": true,
                "elementType": "short",
                "type": "array"
              }
            }
          ],
          "type": "struct"
        }
        >>> print(schema_to_pretty_json(MapType(StringType(), StringType(), True)))
        {
          "keyType": "string",
          "type": "map",
          "valueContainsNull": true,
          "valueType": "string"
        }

    :param schema:
    :return:
    """
    schema_dict = json.loads(schema.json())
    return json.dumps(schema_dict, indent=2, sort_keys=True)


def schema_to_simple_string(schema: DataType) -> str:
    """Convert the given datatype into a simple sql string.
    This method is equivalent to calling [`schema.simpleString()`][pyspark.sql.types.DataType.simpleString] directly.

    Args:
        schema: A DataFrame schema.

    Returns:
        A simpleString representing the DataFrame schema.

    Examples:

        >>> from pyspark.sql.types import *
        >>> schema_to_simple_string(IntegerType())
        'int'
        >>> schema_to_simple_string(StructType([
        ...     StructField('a', ByteType(), True),
        ...     StructField('b', DecimalType(16,8), True)
        ... ]))
        'struct<a:tinyint,b:decimal(16,8)>'
        >>> schema_to_simple_string(StructType([
        ...     StructField('a', DoubleType(), True),
        ...     StructField('b', StringType(), True)
        ... ]))
        'struct<a:double,b:string>'
        >>> schema_to_simple_string(StructType([StructField('a', ArrayType(ShortType(), True), True)]))
        'struct<a:array<smallint>>'
        >>> schema_to_simple_string(MapType(StringType(), StringType(), True))
        'map<string,string>'
    """
    return schema.simpleString()
