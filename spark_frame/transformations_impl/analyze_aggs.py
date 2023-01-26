from pyspark.sql import Column
from pyspark.sql import functions as f
from pyspark.sql.types import StructField


def _to_string(col: Column):
    return col.cast("STRING")


def column_number(col: str, struct_field: StructField, col_num: int) -> Column:  # NOSONAR
    return f.lit(col_num).alias("column_number")


def column_name(col: str, struct_field: StructField, col_num: int) -> Column:  # NOSONAR
    return f.lit(struct_field.name).alias("column_name")


def column_type(col: str, struct_field: StructField, col_num: int) -> Column:  # NOSONAR
    return f.lit(struct_field.dataType.typeName().upper()).alias("column_type")


def count(col: str, struct_field: StructField, col_num: int) -> Column:  # NOSONAR
    return f.count(f.lit(1)).alias("count")


def count_distinct(col: str, struct_field: StructField, col_num: int) -> Column:  # NOSONAR
    return f.count_distinct(col).alias("count_distinct")


def count_null(col: str, struct_field: StructField, col_num: int) -> Column:  # NOSONAR
    return (f.count(f.lit(1)) - f.count(col)).alias("count_null")


def min(col: str, struct_field: StructField, col_num: int) -> Column:  # NOSONAR
    return _to_string(f.min(col)).alias("min")


def max(col: str, struct_field: StructField, col_num: int) -> Column:  # NOSONAR
    return _to_string(f.max(col)).alias("max")
