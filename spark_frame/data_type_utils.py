from typing import Dict, List, Optional, cast

from pyspark.sql.types import ArrayType, DataType, StructField, StructType

from spark_frame.utils import assert_true, get_instantiated_spark_session


def find_wider_type_for_two(t1: DataType, t2: DataType) -> Optional[str]:
    """Python wrapper for Spark's TypeCoercion.find_wider_type_for_two:

    Looking for a widened data type of two given data types with some acceptable loss of precision.
    E.g. there is no common type for double and decimal because double's range
    is larger than decimal, and yet decimal is more precise than double, but in
    union we would cast the decimal into double.

    WARNINGS:

    - the result is a simpleString
    - A SparkSession must already be instantiated for this method to work

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

    :param t1:
    :param t2:
    :return:
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


def flatten_schema(schema: StructType, explode: bool) -> StructType:
    """Transform a œ schema into a new schema where all structs have been flattened.
    The field names are kept, with a '.' separator for struct fields.
    If `explode` option is set, arrays are exploded with a '!' separator.

    Args:
        schema: A Spark [DataFrame][pyspark.sql.DataFrame]'s schema
        explode: If set, arrays are exploded and a '!' separator is appended to their name.

    Returns:
        A flattened schema

    Examples:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.createDataFrame(
        ...     [
        ...         (1, {"a": 1, "b": [{"c": 2, "d": 3}], "e": [4, 5]}),
        ...         (2, None),
        ...     ],
        ...     "id INT, s STRUCT<a:INT, b:ARRAY<STRUCT<c:INT, d:INT, e:ARRAY<INT>>>>"
        ... )
        >>> df.schema.simpleString()
        'struct<id:int,s:struct<a:int,b:array<struct<c:int,d:int,e:array<int>>>>>'
        >>> flatten_schema(df.schema, explode=True).simpleString()
        'struct<id:int,s.a:int,s.b!.c:int,s.b!.d:int,s.b!.e!:int>'
        >>> flatten_schema(df.schema, explode=False).simpleString()
        'struct<id:int,s.a:int,s.b:array<struct<c:int,d:int,e:array<int>>>>'
    """

    def flatten_data_type(
        prefix: str, data_type: DataType, is_nullable: bool, metadata: Dict[str, str]
    ) -> List[StructField]:
        if isinstance(data_type, StructType):
            return flatten_struct_type(data_type, is_nullable, prefix + ".")
        elif isinstance(data_type, ArrayType) and explode:
            return flatten_data_type(
                prefix + "!", data_type.elementType, is_nullable or data_type.containsNull, metadata
            )
        else:
            return [StructField(prefix, data_type, is_nullable, metadata)]

    def flatten_struct_type(schema: StructType, previous_nullable: bool = False, prefix: str = "") -> List[StructField]:
        res = []
        for field in schema:
            if isinstance(field.dataType, StructType):
                res += flatten_struct_type(
                    field.dataType, previous_nullable or field.nullable, prefix + field.name + "."
                )
            else:
                res += flatten_data_type(
                    prefix + field.name, field.dataType, previous_nullable or field.nullable, field.metadata
                )
        return res

    return StructType(flatten_struct_type(schema))
