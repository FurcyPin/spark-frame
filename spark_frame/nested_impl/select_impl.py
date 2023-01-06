from typing import Mapping

from pyspark.sql import DataFrame

from spark_frame.nested_impl.package import ColumnTransformation, resolve_nested_fields


# Workaround: This file is temporarily called "select_impl.py" instead of "select.py" to work around a bug in PyCharm.
# https://youtrack.jetbrains.com/issue/PY-58068
def select(df: DataFrame, fields: Mapping[str, ColumnTransformation]) -> DataFrame:
    """Project a set of expressions and returns a new [DataFrame][pyspark.sql.DataFrame].

    This method is similar to the [DataFrame.select][pyspark.sql.DataFrame.select] method, with the extra
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
        A new DataFrame where only the specified field have been selected and the corresponding
        transformations were applied to each of them.

    Examples:

        *Example 1: non-repeated fields*

        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql import functions as f
        >>> from spark_frame import nested
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''SELECT 1 as id, STRUCT(2 as a, 3 as b) as s''')
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

        Transformations on non-repeated fields may be expressed as a string representing a column name,
        a Column expression or None.
        (In this example the column "id" will be dropped because it was not selected)
        >>> new_df = nested.select(df, {
        ...     "s.a": "s.a",                        # Column name (string)
        ...     "s.b": None,                         # None: use to keep a column without having to repeat its name
        ...     "s.c": f.col("s.a") + f.col("s.b")   # Column expression
        ... })
        >>> new_df.printSchema()
        root
         |-- s: struct (nullable = false)
         |    |-- a: integer (nullable = false)
         |    |-- b: integer (nullable = false)
         |    |-- c: integer (nullable = false)
        <BLANKLINE>
        >>> new_df.show()
        +---------+
        |        s|
        +---------+
        |{2, 3, 5}|
        +---------+
        <BLANKLINE>

        *Example 2: repeated fields*

        >>> df = spark.sql('SELECT 1 as id, ARRAY(STRUCT(1 as a, 2 as b), STRUCT(3 as a, 4 as b)) as s')
        >>> nested.print_schema(df)
        root
         |-- id: integer (nullable = false)
         |-- s!.a: integer (nullable = false)
         |-- s!.b: integer (nullable = false)
        <BLANKLINE>
        >>> df.show()
        +---+----------------+
        | id|               s|
        +---+----------------+
        |  1|[{1, 2}, {3, 4}]|
        +---+----------------+
        <BLANKLINE>

        Transformations on repeated fields must be expressed as higher-order
        functions (lambda expressions or named functions).
        The value passed to this function will correspond to the last repeated element.
        >>> df.transform(nested.select, {
        ...     "s!.a": lambda s: s["a"],
        ...     "s!.b": None,
        ...     "s!.c": lambda s: s["a"] + s["b"]
        ... }).show(truncate=False)
        +----------------------+
        |s                     |
        +----------------------+
        |[{1, 2, 3}, {3, 4, 7}]|
        +----------------------+
        <BLANKLINE>

        String and column expressions can be used on repeated fields as well,
        but their value will be repeated multiple times.
        >>> df.transform(nested.select, {
        ...     "id": None,
        ...     "s!.a": "id",
        ...     "s!.b": f.lit(2)
        ... }).show(truncate=False)
        +---+----------------+
        |id |s               |
        +---+----------------+
        |1  |[{1, 2}, {1, 2}]|
        +---+----------------+
        <BLANKLINE>

        *Example 3: field repeated twice*
        >>> df = spark.sql('''
        ...     SELECT
        ...         1 as id,
        ...         ARRAY(STRUCT(ARRAY(1, 2, 3) as e)) as s1,
        ...         ARRAY(STRUCT(ARRAY(4, 5, 6) as e)) as s2
        ... ''')
        >>> nested.print_schema(df)
        root
         |-- id: integer (nullable = false)
         |-- s1!.e!: integer (nullable = false)
         |-- s2!.e!: integer (nullable = false)
        <BLANKLINE>
        >>> df.show()
        +---+-------------+-------------+
        | id|           s1|           s2|
        +---+-------------+-------------+
        |  1|[{[1, 2, 3]}]|[{[4, 5, 6]}]|
        +---+-------------+-------------+
        <BLANKLINE>

        Here, the lambda expression will be applied to the last repeated element `e`.
        >>> new_df = df.transform(nested.select, {
        ...  "s1!.e!": None,
        ...  "s2!.e!": lambda e : e.cast("DOUBLE")
        ... })
        >>> nested.print_schema(new_df)
        root
         |-- s1!.e!: integer (nullable = false)
         |-- s2!.e!: double (nullable = false)
        <BLANKLINE>
        >>> new_df.show()
        +-------------+-------------------+
        |           s1|                 s2|
        +-------------+-------------------+
        |[{[1, 2, 3]}]|[{[4.0, 5.0, 6.0]}]|
        +-------------+-------------------+
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

        >>> new_df = df.transform(nested.select, {
        ...  "id": None,
        ...  "m1%key": lambda key : f.upper(key),
        ...  "m1%value.a": lambda value : value["a"].cast("DOUBLE")
        ... })
        >>> nested.print_schema(new_df)
        root
         |-- id: integer (nullable = false)
         |-- m1%key: string (nullable = false)
         |-- m1%value.a: double (nullable = false)
        <BLANKLINE>
        >>> new_df.show()
        +---+------------+
        | id|          m1|
        +---+------------+
        |  1|{A -> {2.0}}|
        +---+------------+
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
        >>> new_df = df.transform(nested.select, {
        ...  "id": None,
        ...  "s1!.average": None,
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
        ...  "id": None,
        ...  "s1!.average": None,
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
    return df.select(*resolve_nested_fields(fields, starting_level=df))
