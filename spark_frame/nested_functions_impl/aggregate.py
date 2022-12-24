from typing import Callable, List, Optional, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from spark_frame import fp
from spark_frame.conf import REPETITION_MARKER, STRUCT_SEPARATOR
from spark_frame.fp import PrintableFunction, higher_order
from spark_frame.nested_impl.package import _split_string_and_keep_separator, validate_nested_field_names
from spark_frame.utils import StringOrColumn, assert_true


def _split_field_name(field_name: str) -> List[str]:
    """Split field name according to struct separators (`.`) and repetition markers (`!`)

    >>> field_name = "projects!.tasks!.estimate"
    >>> _split_field_name(field_name)
    ['projects', '!', '.', 'tasks', '!', '.', 'estimate']
    """

    def aux():
        current_alias = field_name
        while current_alias is not None and len(current_alias) > 0:
            node_col, child_col = _split_string_and_keep_separator(current_alias, STRUCT_SEPARATOR, REPETITION_MARKER)
            if child_col is not None and node_col == "":
                node_col = child_col[0]
                child_col = child_col[1:]
            yield node_col
            current_alias = child_col

    return list(aux())


def aggregate(
    field_name: str,
    initial_value: StringOrColumn,
    merge: Callable[[Column, Column], Column],
    start: Optional[Callable[[Column], Column]] = None,
    finish: Optional[Callable[[Column], Column]] = None,
    starting_level: Union[Column, DataFrame, None] = None,
) -> Column:
    """Recursively compute an aggregation of all elements in the given repeated field.

    !!! warning "Limitation: Dots, percents, and exclamation marks are not supported in field names"
        Given the syntax used, every method defined in the `spark_frame.nested` module assumes that all field
        names in DataFrames do not contain any dot `.`, percent `%` or exclamation mark `!`.
        This can be worked around using the transformation
        [`spark_frame.transformations.transform_all_field_names`]
        [spark_frame.transformations_impl.transform_all_field_names.transform_all_field_names].

    Args:
        field_name: Name of the repeated field to sum. It may be repeated multiple times.
        initial_value: Name of column or Column expression.
        merge: A binary function `(acc: Column, x: Column[) -> Column` returning an expression
            of the same type as `initial_value`.
        start: An optional unary function `(x: Column) -> Column` that transforms the values to aggregate into the
            same type as `initial_value`.
        finish: An optional unary function `(x: Column) -> Column` used to convert accumulated value into the final
            result.
        starting_level: Nesting level from which the aggregation is started

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
        >>> employee_df.transform(
        ...     nested.select,
        ...     {
        ...         "employee_id": None,
        ...         "name": None,
        ...         "age": None,
        ...         "total_task_estimate": nf.aggregate(
        ...             field_name="projects!.tasks!.estimate",
        ...             initial_value=f.lit(0).cast("BIGINT"),
        ...             merge=lambda acc, x: acc + x
        ...         ),
        ...         "projects!.task_estimate_per_project": lambda project: nf.aggregate(
        ...             field_name="tasks!.estimate",
        ...             initial_value=f.lit(0).cast("BIGINT"),
        ...             merge=lambda acc, x: acc + x,
        ...             starting_level=project,
        ...         ),
        ...     },
        ... ).show(truncate=False)
        +-----------+----------+---+-------------------+------------+
        |employee_id|name      |age|total_task_estimate|projects    |
        +-----------+----------+---+-------------------+------------+
        |1          |John Smith|30 |29                 |[{13}, {16}]|
        |1          |Jane Doe  |25 |46                 |[{33}, {13}]|
        +-----------+----------+---+-------------------+------------+
        <BLANKLINE>
    """
    validate_nested_field_names(field_name, allow_maps=False)
    agg_merge = PrintableFunction(
        lambda a: f.aggregate(a, initial_value, merge),
        lambda s: f"f.aggregate({s}, initial_value, merge)",
    )
    if finish is not None:
        agg_finish = PrintableFunction(
            lambda a: f.aggregate(f.array(a), initial_value, merge, finish),
            lambda s: f"f.aggregate(f.array({s}), initial_value, merge, finish))",
        )
    else:
        agg_finish = higher_order.identity
    if start is not None:
        agg_start = PrintableFunction(start, lambda s: f"start({s})")
    else:
        agg_start = higher_order.identity

    field_parts = _split_field_name(field_name)

    def recurse_item(parts: List[str], prefix=""):
        key = parts[0]
        is_struct = key == STRUCT_SEPARATOR
        is_repeated = key == REPETITION_MARKER
        has_children = len(parts) > 1
        if has_children:
            child_transformation = recurse_item(parts[1:], prefix + key)
        else:
            child_transformation = agg_start
        if is_struct:
            assert_true(has_children, "Error, this should not happen: struct without children")
            return child_transformation
        elif is_repeated:
            return fp.compose(agg_merge, higher_order.transform(child_transformation))
        else:
            return fp.compose(child_transformation, higher_order.struct_get(key))

    root_transformation = recurse_item(field_parts)
    root_transformation = fp.compose(agg_finish, root_transformation)
    return root_transformation(starting_level)


def _get_sample_data():
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as f

    from spark_frame.schema_utils import schema_from_json

    spark = SparkSession.builder.appName("doctest").getOrCreate()
    json_str = """
    {
      "employees": [
        {
          "employee_id": 1,
          "name": "John Smith",
          "age": 30,
          "projects": [
            {
              "name": "Project A",
              "client": "Acme Inc",
              "tasks": [
                {
                  "name": "Task 1",
                  "estimate": 8
                },
                {
                  "name": "Task 2",
                  "estimate": 5
                }
              ]
            },
            {
              "name": "Project B",
              "client": "Beta Corp",
              "tasks": [
                {
                  "name": "Task 3",
                  "estimate": 13
                },
                {
                  "name": "Task 4",
                  "estimate": 3
                }
              ]
            }
          ]
        },
        {
          "employee_id": 1,
          "name": "Jane Doe",
          "age": 25,
          "projects": [
            {
              "name": "Project C",
              "client": "Gamma Inc",
              "tasks": [
                {
                  "name": "Task 5",
                  "estimate": 20
                },
                {
                  "name": "Task 6",
                  "estimate": 13
                }
              ]
            },
            {
              "name": "Project D",
              "client": "Delta Ltd",
              "tasks": [
                {
                  "name": "Task 7",
                  "estimate": 8
                },
                {
                  "name": "Task 8",
                  "estimate": 5
                }
              ]
            }
          ]
        }
      ]
    }
    """
    raw_df = spark.createDataFrame([(json_str,)], "value STRING").repartition(1)
    json_schema = """
    {
      "fields": [
        {
          "name": "employees",
          "type": {
            "type": "array",
            "containsNull": false,
            "elementType": {
              "type": "struct",
              "fields": [
                {
                  "name": "employee_id",
                  "type": "integer",
                  "nullable": true,
                  "metadata": {}
                },
                {
                  "name": "name",
                  "type": "string",
                  "nullable": true,
                  "metadata": {}
                },
                {
                  "name": "age",
                  "type": "long",
                  "nullable": true,
                  "metadata": {}
                },
                {
                  "name": "projects",
                  "type": {
                    "type": "array",
                    "containsNull": false,
                    "elementType": {
                      "type": "struct",
                      "fields": [
                        {
                          "name": "name",
                          "type": "string",
                          "nullable": true,
                          "metadata": {}
                        },
                        {
                          "name": "client",
                          "type": "string",
                          "nullable": true,
                          "metadata": {}
                        },
                        {
                          "name": "tasks",
                          "type": {
                            "type": "array",
                            "containsNull": false,
                            "elementType": {
                              "type": "struct",
                              "fields": [
                                {
                                  "name": "name",
                                  "type": "string",
                                  "nullable": true,
                                  "metadata": {}
                                },
                                {
                                  "name": "estimate",
                                  "type": "long",
                                  "nullable": true,
                                  "metadata": {}
                                }
                              ]
                            }
                          },
                          "nullable": true,
                          "metadata": {}
                        }
                      ]
                    }
                  },
                  "nullable": true,
                  "metadata": {}
                }
              ]
            }
          },
          "nullable": true,
          "metadata": {}
        }
      ]
    }
    """
    schema = schema_from_json(json_schema)
    df = raw_df.withColumn("value", f.from_json(f.col("value"), schema)).select("value.*")
    employee_df = df.select(f.explode("employees").alias("value")).select("value.*")
    return employee_df
