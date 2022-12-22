from typing import Mapping

from pyspark.sql import DataFrame

from spark_frame import nested
from spark_frame.nested_impl.package import AnyKindOfTransformation, resolve_nested_fields


def with_fields(df: DataFrame, fields: Mapping[str, AnyKindOfTransformation]) -> DataFrame:
    """Return a new [DataFrame][pyspark.sql.DataFrame] by adding or replacing (when they already exist) columns.

    This method is similar to the [DataFrame.withColumn][pyspark.sql.DataFrame.withColumn] method, with the extra
    capability of working on nested and repeated fields (structs and arrays).

    The syntax for field names works as follows:

    - "." is the separator for struct elements
    - "!" must be appended at the end of fields that are repeated

    The following types of transformation are allowed:

    - String and column expressions can be used on any non-repeated field, even nested ones.
    - When working on repeated fields, transformations must be expressed as higher order functions
      (e.g. lambda expressions)
    - `None` can also be used to represent the identity transformation, this is useful to select a field without
       changing and without having to repeat its name.

    Args:
        df: A Spark DataFrame
        fields: A Dict(field_name, transformation_to_apply)

    Returns:
        A new DataFrame with the same fields as the input DataFrame, where the specified transformations have been
        applied to the corresponding fields. If a field name did not exist in the input DataFrame,
        it will be added to the output DataFrame. If it did exist, the original value will be replaced with the new one.

    Examples:

        *Example 1: non-repeated fields*
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql import functions as f
        >>> from spark_frame import nested
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''SELECT 1 as id, STRUCT(2 as a, 3 as b) as s''')
        >>> nested.print_schema(df)
        root
         |-- id: integer (nullable = false)
         |-- s.a: integer (nullable = false)
         |-- s.b: integer (nullable = false)
        <BLANKLINE>
        >>> df.show()
        +---+------+
        | id|     s|
        +---+------+
        |  1|{2, 3}|
        +---+------+
        <BLANKLINE>

        Transformations on non-repeated fields may be expressed as a string representing a column name
        or a Column expression.
        >>> new_df = nested.with_fields(df, {
        ...     "s.id": "id",                                 # column name (string)
        ...     "s.c": f.col("s.a") + f.col("s.b")            # Column expression
        ... })
        >>> new_df.printSchema()
        root
         |-- id: integer (nullable = false)
         |-- s: struct (nullable = false)
         |    |-- a: integer (nullable = false)
         |    |-- b: integer (nullable = false)
         |    |-- id: integer (nullable = false)
         |    |-- c: integer (nullable = false)
        <BLANKLINE>
        >>> new_df.show()
        +---+------------+
        | id|           s|
        +---+------------+
        |  1|{2, 3, 1, 5}|
        +---+------------+
        <BLANKLINE>

        *Example 2: repeated fields*
        >>> df = spark.sql('''
        ...     SELECT
        ...         1 as id,
        ...         ARRAY(STRUCT(1 as a, STRUCT(2 as c) as b), STRUCT(3 as a, STRUCT(4 as c) as b)) as s
        ... ''')
        >>> nested.print_schema(df)
        root
         |-- id: integer (nullable = false)
         |-- s!.a: integer (nullable = false)
         |-- s!.b.c: integer (nullable = false)
        <BLANKLINE>
        >>> df.show()
        +---+--------------------+
        | id|                   s|
        +---+--------------------+
        |  1|[{1, {2}}, {3, {4}}]|
        +---+--------------------+
        <BLANKLINE>

        Transformations on repeated fields must be expressed as
        higher-order functions (lambda expressions or named functions).
        The value passed to this function will correspond to the last repeated element.
        >>> new_df = df.transform(nested.with_fields, {
        ...     "s!.b.d": lambda s: s["a"] + s["b"]["c"]}
        ... )
        >>> nested.print_schema(new_df)
        root
         |-- id: integer (nullable = false)
         |-- s!.a: integer (nullable = false)
         |-- s!.b.c: integer (nullable = false)
         |-- s!.b.d: integer (nullable = false)
        <BLANKLINE>
        >>> new_df.show(truncate=False)
        +---+--------------------------+
        |id |s                         |
        +---+--------------------------+
        |1  |[{1, {2, 3}}, {3, {4, 7}}]|
        +---+--------------------------+
        <BLANKLINE>

        *Example 3: field repeated twice*
        >>> df = spark.sql('SELECT 1 as id, ARRAY(STRUCT(ARRAY(1, 2, 3) as e)) as s')
        >>> nested.print_schema(df)
        root
         |-- id: integer (nullable = false)
         |-- s!.e!: integer (nullable = false)
        <BLANKLINE>
        >>> df.show()
        +---+-------------+
        | id|            s|
        +---+-------------+
        |  1|[{[1, 2, 3]}]|
        +---+-------------+
        <BLANKLINE>

        Here, the lambda expression will be applied to the last repeated element `e`.
        >>> df.transform(nested.with_fields, {"s!.e!": lambda e : e.cast("DOUBLE")}).show()
        +---+-------------------+
        | id|                  s|
        +---+-------------------+
        |  1|[{[1.0, 2.0, 3.0]}]|
        +---+-------------------+
        <BLANKLINE>
    """
    default_columns = {field: None for field in nested.fields(df)}
    fields = {**default_columns, **fields}
    return df.select(*resolve_nested_fields(fields, starting_level=df))
