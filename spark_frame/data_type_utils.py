from typing import Dict, Generator, List, Optional, Tuple, cast

from pyspark.sql.types import ArrayType, DataType, MapType, StructField, StructType

from spark_frame.conf import MAP_KEY, MAP_MARKER, MAP_VALUE, REPETITION_MARKER, STRUCT_SEPARATOR
from spark_frame.utils import assert_true, get_instantiated_spark_session


def is_repeated(schema_field: StructField) -> bool:
    """
    >>> from pyspark.sql.types import IntegerType
    >>> is_repeated(StructField("i", IntegerType()))
    False
    >>> is_repeated(StructField("a", ArrayType(IntegerType())))
    True
    """
    return isinstance(schema_field.dataType, ArrayType)


def is_struct(schema_field: StructField) -> bool:
    """
    >>> from pyspark.sql.types import IntegerType
    >>> is_struct(StructField("i", IntegerType()))
    False
    >>> is_struct(StructField("s", StructType([StructField("i", IntegerType())])))
    True
    """
    return isinstance(schema_field.dataType, StructType)


def is_nullable(schema_field: StructField) -> bool:
    """
    >>> from pyspark.sql.types import IntegerType
    >>> is_nullable(StructField("i", IntegerType()))
    True
    >>> is_nullable(StructField("i", IntegerType(), nullable=False))
    False
    """
    return schema_field.nullable


def find_common_type_for_fields(left_field: StructField, right_field: StructField):
    if is_repeated(right_field) != is_repeated(left_field):
        return None
    elif right_field.dataType == left_field.dataType:
        return None
    else:
        return find_wider_type_for_two(left_field.dataType, right_field.dataType)


def get_common_columns(left_schema: StructType, right_schema: StructType) -> List[Tuple[str, Optional[str]]]:
    """Return a list of common Columns between two DataFrame schemas, along with the widest common type for the
    two columns.

    When columns already have the same type or have incompatible types, the type returned is None.

    Args:
        left_schema: A DataFrame schema
        right_schema: Another DataFrame schema with common columns

    Returns:
        A list of Columns

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df1 = spark.sql('''SELECT 'A' as id, CAST(1 as BIGINT) as a, 'a' as b, NULL as c''')
        >>> df2 = spark.sql('''SELECT 'A' as id, CAST(1 as DOUBLE) as a, ARRAY('a') as b, NULL as d''')
        >>> common_cols = get_common_columns(df1.schema, df2.schema)
        >>> common_cols
        [('id', None), ('a', 'double'), ('b', None)]
    """
    left_fields = {field.name: field for field in left_schema}
    right_fields = {field.name: field for field in right_schema}

    def get_columns():
        for name, left_field in left_fields.items():
            if name in right_fields:
                right_field: StructField = right_fields[name]
                yield name, find_common_type_for_fields(left_field, right_field)

    return list(get_columns())


def find_wider_type_for_two(t1: DataType, t2: DataType) -> Optional[str]:
    """Python wrapper for Spark's TypeCoercion.find_wider_type_for_two:

    Looking for a widened data type of two given data types with some acceptable loss of precision.
    E.g. there is no common type for double and decimal because double's range
    is larger than decimal, and yet decimal is more precise than double, but in
    union we would cast the decimal into double.

    !!! Warning

        - the result is a simpleString
        - A SparkSession must already be instantiated for this method to work

    Args:
        t1: a DataType
        t2: a DataType

    Returns:
        a simpleString representing the smallest common type, None if such type does not exist

    Examples:
        >>> from pyspark.sql.types import DecimalType, LongType, DoubleType, IntegerType
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> find_wider_type_for_two(DecimalType(15, 5), DecimalType(15, 6))
        'decimal(16,6)'
        >>> find_wider_type_for_two(DecimalType(15, 5), DoubleType())
        'double'
        >>> find_wider_type_for_two(LongType(), IntegerType())
        'bigint'
        >>> find_wider_type_for_two(ArrayType(IntegerType()), IntegerType())
    """
    from pyspark.sql import SparkSession

    spark = get_instantiated_spark_session()
    assert_true(spark is not None)
    spark = cast(SparkSession, spark)
    sc = spark.sparkContext

    def _to_java_type(t: DataType):
        return getattr(spark, "_jsparkSession").parseDataType(t.json())

    jt1 = _to_java_type(t1)
    jt2 = _to_java_type(t2)
    j_type_coercion = getattr(
        getattr(getattr(sc, "_jvm").org.apache.spark.sql.catalyst.analysis, "TypeCoercion$"), "MODULE$"
    )
    wider_type = j_type_coercion.findWiderTypeForTwo(jt1, jt2)
    if wider_type.nonEmpty():
        return wider_type.get().simpleString()
    else:
        return None


def flatten_schema(
    schema: StructType,
    explode: bool,
    struct_separator: str = STRUCT_SEPARATOR,
    repetition_marker: str = REPETITION_MARKER,
    map_marker: str = MAP_MARKER,
    map_key: str = MAP_KEY,
    map_value: str = MAP_VALUE,
) -> StructType:
    """Transform a schema into a new schema where all structs have been flattened.
    The field names are kept, with a '.' separator for struct fields.
    If `explode` option is set, arrays and maps are exploded as well.

    - Array names are appended with a '!' repetition marker
    - Map names are appended with '%key' for their keys and '%value' for their values


    Args:
        schema: A Spark [DataFrame][pyspark.sql.DataFrame]'s schema
        explode: If set, arrays and maps are exploded.
        struct_separator: String used to separate structs from their field
        repetition_marker: String used to mark repeated fields
        map_marker: String used to mark maps
        map_key: String used to represent map keys
        map_value: String used to represent map values

    Returns:
        A flattened schema

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('SELECT 1 as id, STRUCT(2 as a, ARRAY(STRUCT(3 as c, 4 as d, ARRAY(5) as e)) as b) as s')
        >>> df.schema.simpleString()
        'struct<id:int,s:struct<a:int,b:array<struct<c:int,d:int,e:array<int>>>>>'
        >>> flatten_schema(df.schema, explode=True).simpleString()
        'struct<id:int,s.a:int,s.b!.c:int,s.b!.d:int,s.b!.e!:int>'
        >>> flatten_schema(df.schema, explode=False).simpleString()
        'struct<id:int,s.a:int,s.b:array<struct<c:int,d:int,e:array<int>>>>'

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('SELECT 1 as id, MAP(STRUCT(2 as a), STRUCT(3 as b)) as m')
        >>> df.schema.simpleString()
        'struct<id:int,m:map<struct<a:int>,struct<b:int>>>'
        >>> flatten_schema(df.schema, explode=True).simpleString()
        'struct<id:int,m%key.a:int,m%value.b:int>'
        >>> flatten_schema(df.schema, explode=False).simpleString()
        'struct<id:int,m:map<struct<a:int>,struct<b:int>>>'
    """

    def flatten_data_type(
        data_type: DataType, is_nullable: bool, metadata: Dict[str, str], prefix: str
    ) -> Generator[StructField, None, None]:
        if isinstance(data_type, StructType):
            yield from flatten_struct_type(data_type, is_nullable, prefix + struct_separator)
        elif isinstance(data_type, ArrayType) and explode:
            yield from flatten_data_type(
                data_type.elementType, is_nullable or data_type.containsNull, metadata, prefix + repetition_marker
            )
        elif isinstance(data_type, MapType) and explode:
            yield from flatten_data_type(data_type.keyType, is_nullable, metadata, prefix + map_marker + map_key)
            yield from flatten_data_type(
                data_type.valueType,
                is_nullable or data_type.valueContainsNull,
                metadata,
                prefix + map_marker + map_value,
            )
        else:
            yield StructField(prefix, data_type, is_nullable, metadata)

    def flatten_struct_type(
        schema: StructType, previous_nullable: bool = False, prefix: str = ""
    ) -> Generator[StructField, None, None]:
        for field in schema:
            yield from flatten_data_type(
                field.dataType, previous_nullable or is_nullable(field), field.metadata, prefix + field.name
            )

    return StructType(list(flatten_struct_type(schema)))
