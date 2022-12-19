from pyspark.sql import SparkSession


def _get_sample_data():
    spark = SparkSession.builder.appName("doctest").getOrCreate()
    df = spark.sql(
        """
        SELECT
            1 as id,
            STRUCT("Bulbasaur" as english, "Bulbizarre" as french) as name,
            ARRAY("Grass", "Poison") as types
    """
    )
    return df


def fields():
    """First, let's distinguish the notion of `Column` and `Field`.
    Both terms are already used in Spark, but we chose here to make the following distinction:

    - A `Column` is a root-level column of a DataFrame.
    - A `Field` is any column or sub-column inside a struct of the DataFrame.

    Examples: Example: let's consider the following DataFrame

        >>> from spark_frame.examples.reference_nested import _get_sample_data
        >>> df = _get_sample_data()
        >>> df.show(truncate=False)  # noqa: E501 # doctest: +NORMALIZE_WHITESPACE
        +---+-----------------------+---------------+
        |id |name                   |types          |
        +---+-----------------------+---------------+
        |1  |{Bulbasaur, Bulbizarre}|[Grass, Poison]|
        +---+-----------------------+---------------+
        <BLANKLINE>
        >>> df.printSchema()
        root
         |-- id: integer (nullable = false)
         |-- name: struct (nullable = false)
         |    |-- english: string (nullable = false)
         |    |-- french: string (nullable = false)
         |-- types: array (nullable = false)
         |    |-- element: string (containsNull = false)
        <BLANKLINE>

        This DataFrame has 3 columns:

        ```
        id
        name
        types
        ```

        But it has 4 fields:

        ```
        id
        name.english
        name.french
        types!
        ```

        This can be seen by using the method
        [spark_frame.nested.print_schema](/reference/#spark_framenested.print_schema)

        >>> from spark_frame import nested
        >>> nested.print_schema(df)
        root
         |-- id: integer (nullable = false)
         |-- name.english: string (nullable = false)
         |-- name.french: string (nullable = false)
         |-- types!: string (nullable = false)
        <BLANKLINE>

        As we can see, some field names contain dots (`.`) or exclamation marks (`!`), they convey the following
        meaning:

        - A dot (`.`) represents a struct.
        - An exclamation mark (`!`) represents an array.

        While the "dot" syntax for "structs" should feel familiar to users, the exclamation mark (`!`) should feel new.
        It is used as a *repetition marker* indicating that this field is repeated.

        !!! tip "Tip"
            It is important to not forget to use exclamation marks (`!`) when mentionning a field.
            For instance:

            - `types` designates the root-level field which is of type `ARRAY<STRING>`
            - `types!` designates the elements inside this array

            In particular, if a field `"my_field"` is of type `ARRAY<ARRAY<STRING>>`, the innermost elements of the
            arrays will be designated as `"my_field!!"` with two exclamation marks.

        !!! warning "Limitation: Do not use dots or exclamation marks in field names"
            Given the syntax used, every method defined in the `spark_frame.nested` module assumes that all field
            names in DataFrames do not contain any dot `.` or exclamation mark `!`. We will work on addressing this
            limitation in further developments.

        !!! warning "Limitation: Do not use Maps"
            Fields of type `Map<K,V>` are not currently supported and flattened.
            We recommended to use [spark_frame.transformations.convert_all_maps_to_arrays](
            /reference/#spark_frame.transformations_impl.convert_all_maps_to_arrays.convert_all_maps_to_arrays)
            to automatically cast all the maps of a DataFrame into `ARRAY<STRUCT<key, value>>`.
    """
    # This is a hacky way to have doctests that runs in the pipeline and are usable in the doc thanks to mkdocstrings
