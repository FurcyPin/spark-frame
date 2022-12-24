from typing import Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from spark_frame.nested_functions_impl.aggregate import aggregate


def sum(field_name: str, starting_level: Union[Column, DataFrame, None] = None) -> Column:
    """Recursively compute the sum of all elements in the given repeated field.

    !!! warning "Limitation: Dots, percents, and exclamation marks are not supported in field names"
        Given the syntax used, every method defined in the `spark_frame.nested` module assumes that all field
        names in DataFrames do not contain any dot `.`, percent `%` or exclamation mark `!`.
        This can be worked around using the transformation
        [`spark_frame.transformations.transform_all_field_names`]
        [spark_frame.transformations_impl.transform_all_field_names.transform_all_field_names].

    Args:
        field_name: Name of the repeated field to sum. It may be repeated multiple times.
        starting_level: Nesting level from which the aggregation is started.

    Returns:
        A Column expression

    Examples:
        *Example 1*
        >>> from spark_frame.nested_functions_impl.aggregate import _get_sample_data
        >>> from spark_frame import nested
        >>> from spark_frame import nested_functions as nf
        >>> employee_df = _get_sample_data()
        >>> nested.print_schema(employee_df)
        root
         |-- employee_id: integer (nullable = true)
         |-- name: string (nullable = true)
         |-- age: long (nullable = true)
         |-- projects!.name: string (nullable = true)
         |-- projects!.client: string (nullable = true)
         |-- projects!.tasks!.name: string (nullable = true)
         |-- projects!.tasks!.estimate: long (nullable = true)
        <BLANKLINE>
        >>> employee_df.withColumn("projects", f.to_json("projects.tasks")).show(truncate=False)  # noqa: E501
        +-----------+----------+---+-----------------------------------------------------------------------------------------------------------------------------------+
        |employee_id|name      |age|projects                                                                                                                           |
        +-----------+----------+---+-----------------------------------------------------------------------------------------------------------------------------------+
        |1          |John Smith|30 |[[{"name":"Task 1","estimate":8},{"name":"Task 2","estimate":5}],[{"name":"Task 3","estimate":13},{"name":"Task 4","estimate":3}]] |
        |1          |Jane Doe  |25 |[[{"name":"Task 5","estimate":20},{"name":"Task 6","estimate":13}],[{"name":"Task 7","estimate":8},{"name":"Task 8","estimate":5}]]|
        +-----------+----------+---+-----------------------------------------------------------------------------------------------------------------------------------+
        <BLANKLINE>
        >>> employee_df.transform(nested.select, {
        ...     "employee_id": None,
        ...     "name": None,
        ...     "age": None,
        ...     "projects!.tasks!.estimate": None
        ... }).show(truncate=False)
        +-----------+----------+---+------------------------------+
        |employee_id|name      |age|projects                      |
        +-----------+----------+---+------------------------------+
        |1          |John Smith|30 |[{[{8}, {5}]}, {[{13}, {3}]}] |
        |1          |Jane Doe  |25 |[{[{20}, {13}]}, {[{8}, {5}]}]|
        +-----------+----------+---+------------------------------+
        <BLANKLINE>
        >>> employee_df.transform(nested.select, {
        ...     "employee_id": None,
        ...     "name": None,
        ...     "age": None,
        ...     "total_task_estimate": nf.sum("projects!.tasks!.estimate"),
        ...     "projects!.task_estimate_per_project": lambda project: nf.sum("tasks!.estimate", starting_level=project),
        ... }).show(truncate=False)
        +-----------+----------+---+-------------------+------------+
        |employee_id|name      |age|total_task_estimate|projects    |
        +-----------+----------+---+-------------------+------------+
        |1          |John Smith|30 |29                 |[{13}, {16}]|
        |1          |Jane Doe  |25 |46                 |[{33}, {13}]|
        +-----------+----------+---+-------------------+------------+
        <BLANKLINE>

        *Example 2 : with all kind of nested structures*
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''SELECT
        ...     1 as id,
        ...     ARRAY(STRUCT(1 as a), STRUCT(2 as a)) as s1,
        ...     ARRAY(ARRAY(1, 2), ARRAY(3, 4)) as s2,
        ...     ARRAY(ARRAY(STRUCT(1 as a)), ARRAY(STRUCT(2 as a))) as s3,
        ...     ARRAY(STRUCT(ARRAY(1, 2) as a), STRUCT(ARRAY(3, 4) as a)) as s4,
        ...     ARRAY(
        ...         STRUCT(ARRAY(STRUCT(STRUCT(1 as c) as b), STRUCT(STRUCT(2 as c) as b)) as a),
        ...         STRUCT(ARRAY(STRUCT(STRUCT(3 as c) as b), STRUCT(STRUCT(4 as c) as b)) as a)
        ...     ) as s5
        ... ''')
        >>> nested.print_schema(df)
        root
         |-- id: integer (nullable = false)
         |-- s1!.a: integer (nullable = false)
         |-- s2!!: integer (nullable = false)
         |-- s3!!.a: integer (nullable = false)
         |-- s4!.a!: integer (nullable = false)
         |-- s5!.a!.b.c: integer (nullable = false)
        <BLANKLINE>
        >>> df.show(truncate=False)
        +---+----------+----------------+--------------+--------------------+------------------------------------+
        |id |s1        |s2              |s3            |s4                  |s5                                  |
        +---+----------+----------------+--------------+--------------------+------------------------------------+
        |1  |[{1}, {2}]|[[1, 2], [3, 4]]|[[{1}], [{2}]]|[{[1, 2]}, {[3, 4]}]|[{[{{1}}, {{2}}]}, {[{{3}}, {{4}}]}]|
        +---+----------+----------------+--------------+--------------------+------------------------------------+
        <BLANKLINE>
        >>> df.select(nf.sum("s1!.a").alias("sum")).show()
        +---+
        |sum|
        +---+
        |  3|
        +---+
        <BLANKLINE>
        >>> df.select(nf.sum("s2!!").alias("sum")).show()
        +---+
        |sum|
        +---+
        | 10|
        +---+
        <BLANKLINE>
        >>> df.select(nf.sum("s3!!.a").alias("sum")).show()
        +---+
        |sum|
        +---+
        |  3|
        +---+
        <BLANKLINE>
        >>> df.select(nf.sum("s4!.a!").alias("sum")).show()
        +---+
        |sum|
        +---+
        | 10|
        +---+
        <BLANKLINE>
        >>> df.select(nf.sum("s5!.a!.b.c").alias("sum")).show()
        +---+
        |sum|
        +---+
        | 10|
        +---+
        <BLANKLINE>
    """
    initial_value = f.lit(0).cast("BIGINT")

    def merge(acc: Column, x: Column) -> Column:
        return acc + x

    return aggregate(field_name, initial_value=initial_value, merge=merge, starting_level=starting_level)
