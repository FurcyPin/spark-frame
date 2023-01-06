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
    - "!" must be appended at the end of fields that are repeated (arrays)
    - Map keys are appended with `%key`
    - Map values are appended with `%value`

    The following types of transformation are allowed:

    - String and column expressions can be used on any non-repeated field, even nested ones.
    - When working on repeated fields, transformations must be expressed as higher order functions
      (e.g. lambda expressions). String and column expressions can be used on repeated fields as well,
      but their value will be repeated multiple times.
    - When working on multiple levels of nested arrays, higher order functions may take multiple arguments,
      corresponding to each level of repetition (See Example 5.).
    - `None` can also be used to represent the identity transformation, this is useful to select a field without
       changing and without having to repeat its name.

    !!! warning "Limitation: Dots, percents, and exclamation marks are not supported in field names"
        Given the syntax used, every method defined in the `spark_frame.nested` module assumes that all field
        names in DataFrames do not contain any dot `.`, percent `%` or exclamation mark `!`.
        This can be worked around using the transformation
        [`spark_frame.transformations.transform_all_field_names`]
        [spark_frame.transformations_impl.transform_all_field_names.transform_all_field_names].

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

        String and column expressions can be used on repeated fields as well,
        but their value will be repeated multiple times.
        >>> df.transform(nested.with_fields, {
        ...     "id": None,
        ...     "s!.a": "id",
        ...     "s!.b.c": f.lit(2)
        ... }).show(truncate=False)
        +---+--------------------+
        |id |s                   |
        +---+--------------------+
        |1  |[{1, {2}}, {1, {2}}]|
        +---+--------------------+
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

        *Example 4: Dataframe with maps*
        >>> df = spark.sql('''
        ...     SELECT
        ...         1 as id,
        ...         MAP("a", STRUCT(2 as a, 3 as b)) as m1
        ... ''')
        >>> nested.print_schema(df)
        root
         |-- id: integer (nullable = false)
         |-- m1%key: string (nullable = false)
         |-- m1%value.a: integer (nullable = false)
         |-- m1%value.b: integer (nullable = false)
        <BLANKLINE>
        >>> df.show()
        +---+-------------+
        | id|           m1|
        +---+-------------+
        |  1|{a -> {2, 3}}|
        +---+-------------+
        <BLANKLINE>

        >>> new_df = df.transform(nested.with_fields, {
        ...  "m1%key": lambda key : f.upper(key),
        ...  "m1%value.a": lambda value : value["a"].cast("DOUBLE")
        ... })
        >>> nested.print_schema(new_df)
        root
         |-- id: integer (nullable = false)
         |-- m1%key: string (nullable = false)
         |-- m1%value.a: double (nullable = false)
         |-- m1%value.b: integer (nullable = false)
        <BLANKLINE>
        >>> new_df.show()
        +---+---------------+
        | id|             m1|
        +---+---------------+
        |  1|{A -> {2.0, 3}}|
        +---+---------------+
        <BLANKLINE>

        *Example 5: Accessing multiple repetition levels*
        >>> df = spark.sql('''
        ...     SELECT
        ...         1 as id,
        ...         ARRAY(
        ...             STRUCT(2 as average, ARRAY(1, 2, 3) as values),
        ...             STRUCT(3 as average, ARRAY(1, 2, 3, 4, 5) as values)
        ...         ) as s1
        ... ''')
        >>> nested.print_schema(df)
        root
         |-- id: integer (nullable = false)
         |-- s1!.average: integer (nullable = false)
         |-- s1!.values!: integer (nullable = false)
        <BLANKLINE>
        >>> df.show(truncate=False)
        +---+--------------------------------------+
        |id |s1                                    |
        +---+--------------------------------------+
        |1  |[{2, [1, 2, 3]}, {3, [1, 2, 3, 4, 5]}]|
        +---+--------------------------------------+
        <BLANKLINE>

        Here, the transformation applied to "s1!.values!" takes two arguments.
        >>> new_df = df.transform(nested.with_fields, {
        ...  "s1!.values!": lambda s1, value : value - s1["average"]
        ... })
        >>> new_df.show(truncate=False)
        +---+-----------------------------------------+
        |id |s1                                       |
        +---+-----------------------------------------+
        |1  |[{2, [-1, 0, 1]}, {3, [-2, -1, 0, 1, 2]}]|
        +---+-----------------------------------------+
        <BLANKLINE>

        Extra arguments can be added to the left for each repetition level, up to the root level.
        >>> new_df = df.transform(nested.with_fields, {
        ...  "s1!.values!": lambda root, s1, value : value - s1["average"] + root["id"]
        ... })
        >>> new_df.show(truncate=False)
        +---+---------------------------------------+
        |id |s1                                     |
        +---+---------------------------------------+
        |1  |[{2, [0, 1, 2]}, {3, [-1, 0, 1, 2, 3]}]|
        +---+---------------------------------------+
        <BLANKLINE>

    """
    default_columns = {field: None for field in nested.fields(df)}
    fields = {**default_columns, **fields}
    return df.select(*resolve_nested_fields(fields, starting_level=df))
