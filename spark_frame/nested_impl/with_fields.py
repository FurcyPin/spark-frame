from typing import Mapping

from pyspark.sql import DataFrame

from spark_frame import nested
from spark_frame.conf import REPETITION_MARKER, STRUCT_SEPARATOR
from spark_frame.fp import PrintableFunction, higher_order
from spark_frame.nested_impl.package import ColumnTransformation, resolve_nested_fields


def with_fields(df: DataFrame, fields: Mapping[str, ColumnTransformation]) -> DataFrame:
    """Return a new [DataFrame](pyspark.sql.DataFrame) by adding or replacing (when they already exist) columns.

    This method is similar to the [DataFrame.withColumn](pyspark.sql.DataFrame.withColumn) method, with the extra
    capability of working on nested and repeated fields (structs and arrays).

    The syntax for column names works as follows:
    - "." is the separator for struct elements
    - "!" must be appended at the end of fields that are repeated

    The following types of transformation are allowed:
    - String and column expressions can be used on any non-repeated field, even nested ones.
    - When working on repeated fields, transformations must be expressed as higher order functions
      (e.g. lambda expressions)

    Args:
        df: A Spark DataFrame
        fields: A Dict(field_name, transformation_to_apply)

    Returns:
        A new DataFrame with the same fields as the input DataFrame, where the specified transformations have been
        applied to the corresponding fields. If a field name did not exist in the input DataFrame,
        it will be added to the output DataFrame. If it did exist, the original value will be replaced with the new one.

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql import functions as f
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''
        ...  SELECT INLINE(ARRAY(
        ...    STRUCT(1 as id, STRUCT(2 as a, 3 as b) as s)
        ...  ))
        ... ''')
        >>> df.printSchema()
        root
         |-- id: integer (nullable = false)
         |-- s: struct (nullable = false)
         |    |-- a: integer (nullable = false)
         |    |-- b: integer (nullable = false)
        <BLANKLINE>
        >>> df.show()
        +---+------+
        | id|     s|
        +---+------+
        |  1|{2, 3}|
        +---+------+
        <BLANKLINE>

        >>> new_df = with_fields(df, {
        ...     "id": "id",
        ...     "s.c": f.col("s.a") + f.col("s.b")
        ... })
        >>> new_df.printSchema()
        root
         |-- id: integer (nullable = false)
         |-- s: struct (nullable = false)
         |    |-- a: integer (nullable = false)
         |    |-- b: integer (nullable = false)
         |    |-- c: integer (nullable = false)
        <BLANKLINE>
        >>> new_df.show()
        +---+---------+
        | id|        s|
        +---+---------+
        |  1|{2, 3, 5}|
        +---+---------+
        <BLANKLINE>

        >>> new_df = with_fields(df, {
        ...     "id": "id",
        ...     "s.c": lambda s: s["a"] + s["b"]
        ... })
        >>> new_df.printSchema()
        root
         |-- id: integer (nullable = false)
         |-- s: struct (nullable = false)
         |    |-- a: integer (nullable = false)
         |    |-- b: integer (nullable = false)
         |    |-- c: integer (nullable = false)
        <BLANKLINE>
        >>> new_df.show()
        +---+---------+
        | id|        s|
        +---+---------+
        |  1|{2, 3, 5}|
        +---+---------+
        <BLANKLINE>

        >>> df = spark.sql('''
        ...  SELECT INLINE(ARRAY(
        ...    STRUCT(1 as id, ARRAY(STRUCT(1 as a, 2 as b), STRUCT(3 as a, 4 as b)) as s)
        ...  ))
        ... ''')
        >>> df.show()
        +---+----------------+
        | id|               s|
        +---+----------------+
        |  1|[{1, 2}, {3, 4}]|
        +---+----------------+
        <BLANKLINE>
        >>> df.transform(with_fields, {"s!.c": lambda s: s["a"] + s["b"]}).show(truncate=False)
        +---+----------------------+
        |id |s                     |
        +---+----------------------+
        |1  |[{1, 2, 3}, {3, 4, 7}]|
        +---+----------------------+
        <BLANKLINE>

        >>> df = spark.sql('''
        ...  SELECT INLINE(ARRAY(
        ...    STRUCT(1 as id, ARRAY(STRUCT(ARRAY(1, 2, 3) as e)) as s)
        ...  ))
        ... ''')
        >>> df.show()
        +---+-------------+
        | id|            s|
        +---+-------------+
        |  1|[{[1, 2, 3]}]|
        +---+-------------+
        <BLANKLINE>
        >>> df.transform(with_fields, {"s!.e!": lambda e : e.cast("DOUBLE")}).show()
        +---+-------------------+
        | id|                  s|
        +---+-------------------+
        |  1|[{[1.0, 2.0, 3.0]}]|
        +---+-------------------+
        <BLANKLINE>
    """

    def identity_for_field(field: str) -> PrintableFunction:
        if field[-1] == REPETITION_MARKER:
            return higher_order.identity
        else:
            field_short_name = field.split(STRUCT_SEPARATOR)[-1]
            return higher_order.safe_struct_get(field_short_name)

    default_columns = {field: identity_for_field(field) for field in nested.fields(df)}
    fields = {**default_columns, **fields}
    return df.select(*resolve_nested_fields(fields))
