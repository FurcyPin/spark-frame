from pyspark.sql import DataFrame

from spark_frame.conf import REPETITION_MARKER, STRUCT_SEPARATOR
from spark_frame.data_type_utils import flatten_schema
from spark_frame.fp import higher_order
from spark_frame.fp.printable_function import PrintableFunction
from spark_frame.nested import resolve_nested_columns


def sort_all_arrays(df: DataFrame) -> DataFrame:
    """Given a DataFrame, sort all fields of type `ARRAY` in a canonical order, making them comparable.
    This also applies to nested fields, even those inside other arrays.

    !!! warning "Limitations"
        - Arrays containing sub-fields of type Map cannot be sorted, as the Map type is not comparable.
        - Fields located inside Maps will not be affected.
        - A possible workaround to these limitations is to use the transformation [`convert_all_maps_to_arrays`](
        /reference/#spark_frame.transformations_impl.convert_all_maps_to_arrays.convert_all_maps_to_arrays)

    Args:
        df: A Spark DataFrame

    Returns:
        A new DataFrame where all arrays have been sorted.

    Examples:

        *Example 1:* with a simple `ARRAY<INT>`

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('SELECT 1 as id, ARRAY(3, 2, 1) as a')
        >>> df.show()
        +---+---------+
        | id|        a|
        +---+---------+
        |  1|[3, 2, 1]|
        +---+---------+
        <BLANKLINE>
        >>> sort_all_arrays(df).show()
        +---+---------+
        | id|        a|
        +---+---------+
        |  1|[1, 2, 3]|
        +---+---------+
        <BLANKLINE>

        *Example 2:* with an `ARRAY<STRUCT<...>>`

        >>> df = spark.sql('SELECT ARRAY(STRUCT(2 as a, 1 as b), STRUCT(1 as a, 2 as b), STRUCT(1 as a, 1 as b)) as s')
        >>> df.show(truncate=False)
        +------------------------+
        |s                       |
        +------------------------+
        |[{2, 1}, {1, 2}, {1, 1}]|
        +------------------------+
        <BLANKLINE>
        >>> df.transform(sort_all_arrays).show(truncate=False)
        +------------------------+
        |s                       |
        +------------------------+
        |[{1, 1}, {1, 2}, {2, 1}]|
        +------------------------+
        <BLANKLINE>

        *Example 3:* with an `ARRAY<STRUCT<STRUCT<...>>>`

        >>> df = spark.sql('''SELECT ARRAY(
        ...         STRUCT(STRUCT(2 as a, 2 as b) as s),
        ...         STRUCT(STRUCT(1 as a, 2 as b) as s)
        ...     ) as l1
        ... ''')
        >>> df.show(truncate=False)
        +--------------------+
        |l1                  |
        +--------------------+
        |[{{2, 2}}, {{1, 2}}]|
        +--------------------+
        <BLANKLINE>
        >>> df.transform(sort_all_arrays).show(truncate=False)
        +--------------------+
        |l1                  |
        +--------------------+
        |[{{1, 2}}, {{2, 2}}]|
        +--------------------+
        <BLANKLINE>

        *Example 4:* with an `ARRAY<ARRAY<ARRAY<INT>>>`

        As this example shows, the innermost arrays are sorted before the outermost arrays.

        >>> df = spark.sql('''SELECT ARRAY(
        ...         ARRAY(ARRAY(4, 1), ARRAY(3, 2)),
        ...         ARRAY(ARRAY(2, 2), ARRAY(2, 1))
        ...     ) as l1
        ... ''')
        >>> df.show(truncate=False)
        +------------------------------------+
        |l1                                  |
        +------------------------------------+
        |[[[4, 1], [3, 2]], [[2, 2], [2, 1]]]|
        +------------------------------------+
        <BLANKLINE>
        >>> df.transform(sort_all_arrays).show(truncate=False)
        +------------------------------------+
        |l1                                  |
        +------------------------------------+
        |[[[1, 2], [2, 2]], [[1, 4], [2, 3]]]|
        +------------------------------------+
        <BLANKLINE>


    """
    schema_flat = flatten_schema(
        df.schema, explode=True, struct_separator=STRUCT_SEPARATOR, repetition_marker=REPETITION_MARKER
    )

    def build_col(col_name: str) -> PrintableFunction:
        is_repeated = col_name[-1] == REPETITION_MARKER
        col = col_name.split(STRUCT_SEPARATOR)[-1]
        if is_repeated:
            return higher_order.identity
        else:
            return higher_order.safe_struct_get(col)

    columns = {field.name: build_col(field.name) for field in schema_flat}
    return df.select(*resolve_nested_columns(columns, sort=True))
