def transform_nested_fields():
    """

    Examples: Let's take a sample DataFrame with a deeply nested schema

        >>> from spark_frame.examples.working_with_nested_data import _get_sample_employee_data
        >>> from pyspark.sql import functions as f
        >>> df = _get_sample_employee_data()
        >>> df.printSchema()
        root
         |-- employee_id: integer (nullable = true)
         |-- name: string (nullable = true)
         |-- age: long (nullable = true)
         |-- skills: array (nullable = true)
         |    |-- element: struct (containsNull = true)
         |    |    |-- name: string (nullable = true)
         |    |    |-- level: string (nullable = true)
         |-- projects: array (nullable = true)
         |    |-- element: struct (containsNull = true)
         |    |    |-- name: string (nullable = true)
         |    |    |-- client: string (nullable = true)
         |    |    |-- tasks: array (nullable = true)
         |    |    |    |-- element: struct (containsNull = true)
         |    |    |    |    |-- name: string (nullable = true)
         |    |    |    |    |-- description: string (nullable = true)
         |    |    |    |    |-- status: string (nullable = true)
         |    |    |    |    |-- estimate: long (nullable = true)
        <BLANKLINE>
        >>> df.show(truncate=False)  # noqa: E501
        +-----------+----------+---+---------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |employee_id|name      |age|skills                                       |projects                                                                                                                                                                                                                          |
        +-----------+----------+---+---------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |1          |John Smith|30 |[{Java, expert}, {Python, intermediate}]     |[{Project A, Acme Inc, [{Task 1, Implement feature X, completed, 8}, {Task 2, Fix bug Y, in progress, 5}]}, {Project B, Beta Corp, [{Task 3, Implement feature Z, pending, 13}, {Task 4, Improve performance, in progress, 3}]}]  |
        |2          |Jane Doe  |25 |[{JavaScript, advanced}, {PHP, intermediate}]|[{Project C, Gamma Inc, [{Task 5, Implement feature W, completed, 20}, {Task 6, Fix bug V, in progress, 13}]}, {Project D, Delta Ltd, [{Task 7, Implement feature U, pending, 8}, {Task 8, Improve performance, in progress, 5}]}]|
        +-----------+----------+---+---------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        <BLANKLINE>

        As we can see, the schema has two top-level columns of type ARRAY (`skills` and `projects`),
        and the `projects` array contains a second level of repetition `projects.tasks`.

        Manipulating this DataFrame with Spark can quickly become really painful, and it is still quite simple
        compare to what engineers may encounter while working with entreprise-grade datasets.

        ### The task
        Let's say we want to enrich this DataFrame by performing the following changes:

        - Change the `skills.level` to uppercase
        - Cast the `projects.tasks.estimate` to double

        ### Without spark_frame.nested
        Prior to Spark 3.0, we would have had only two choices:

        1. Flatten `skills` and `projects.tasks` into two separate DataFrames, perform the transformation then join the
           two DataFrames back together.
        2. Write a custom Python UDF to perform the changes.

        Option 1. is not a good solution, as it would be quite costly, and require several shuffle operations.

        Option 2. is not great either, as the Python UDF would be slow and not very reusable. Had we been using Java
        or Scala, this might have been a better option already, as we would not incure the performance costs associated
        with Python UDFs, but this would still have required a lot of work to code the whole Employee data structure
        in Java/Scala before being able to manipulate it.

        Since Spark 3.1.0, a third option is available, which consists in using [pyspark.sql.functions.transform][]
        and [pyspark.sql.Column.withField][] to achieve our goal.

        However, the code that we need to write is quite complex:

        >>> new_df = df.withColumn(
        ...     "skills",
        ...     f.transform(f.col("skills"), lambda skill: skill.withField("level", f.upper(skill["level"])))
        ... ).withColumn(
        ...     "projects",
        ...     f.transform(
        ...         f.col("projects"),
        ...         lambda project: project.withField(
        ...             "tasks",
        ...             f.transform(
        ...                 project["tasks"],
        ...                 lambda task: task.withField("estimate", task["estimate"].cast("DOUBLE"))),
        ...         ),
        ...     ),
        ... )
        >>> new_df.printSchema()
        root
         |-- employee_id: integer (nullable = true)
         |-- name: string (nullable = true)
         |-- age: long (nullable = true)
         |-- skills: array (nullable = true)
         |    |-- element: struct (containsNull = true)
         |    |    |-- name: string (nullable = true)
         |    |    |-- level: string (nullable = true)
         |-- projects: array (nullable = true)
         |    |-- element: struct (containsNull = true)
         |    |    |-- name: string (nullable = true)
         |    |    |-- client: string (nullable = true)
         |    |    |-- tasks: array (nullable = true)
         |    |    |    |-- element: struct (containsNull = true)
         |    |    |    |    |-- name: string (nullable = true)
         |    |    |    |    |-- description: string (nullable = true)
         |    |    |    |    |-- status: string (nullable = true)
         |    |    |    |    |-- estimate: double (nullable = true)
        <BLANKLINE>
        >>> new_df.select("employee_id", "name", "age", "skills").show(truncate=False)
        +-----------+----------+---+---------------------------------------------+
        |employee_id|name      |age|skills                                       |
        +-----------+----------+---+---------------------------------------------+
        |1          |John Smith|30 |[{Java, EXPERT}, {Python, INTERMEDIATE}]     |
        |2          |Jane Doe  |25 |[{JavaScript, ADVANCED}, {PHP, INTERMEDIATE}]|
        +-----------+----------+---+---------------------------------------------+
        <BLANKLINE>

        As we can see, the transformation worked: the schema is the same except `projects.tasks.estimate` which is now
        a `double`, and `skills.name` is now in uppercase. But hopefully we can agree that the code to achieve this
        looks quite complex, and that it's complexity would grow even more if we tried to perform more transformations
        at the same time.

        ### With spark_frame.nested

        The module [`spark_frame.nested`](/spark-frame/reference/nested) proposes several methods to help us deal
        with nested data structure more easily.
        First, let's use [`spark_frame.nested.print_schema`][spark_frame.nested_impl.print_schema.print_schema] to get
        a flat version of the DataFrame's schema:

        >>> from spark_frame import nested
        >>> nested.print_schema(df)
        root
         |-- employee_id: integer (nullable = true)
         |-- name: string (nullable = true)
         |-- age: long (nullable = true)
         |-- skills!.name: string (nullable = true)
         |-- skills!.level: string (nullable = true)
         |-- projects!.name: string (nullable = true)
         |-- projects!.client: string (nullable = true)
         |-- projects!.tasks!.name: string (nullable = true)
         |-- projects!.tasks!.description: string (nullable = true)
         |-- projects!.tasks!.status: string (nullable = true)
         |-- projects!.tasks!.estimate: long (nullable = true)
        <BLANKLINE>

        As we can see, this is the same schema as before, but instead of being displayed as a tree, it is displayed
        as a flat list where each field is represented with its full name. We can also see that fields of type ARRAY
        can be easily identified thanks to the exclamation marks (`!`) added after their names.
        Once you get used to it, this flat representation is more compact and easier to read than the
        tree representation, while conveying the same amount of information.

        This notation will also help us performing the target transformations more easily.
        As a reminder, we want to:

        - Change the `skills.level` to uppercase
        - Cast the `projects.tasks.estimate` to double

        Using the [`spark_frame.nested.with_fields`]
        [spark_frame.nested_impl.with_fields.with_fields] method, this can be done like this:
        >>> new_df = df.transform(nested.with_fields, {
        ...     "skills!.level": lambda skill: f.upper(skill["level"]),
        ...     "projects!.tasks!.estimate": lambda task: task["estimate"].cast("DOUBLE")
        ... })
        >>> nested.print_schema(new_df)
        root
         |-- employee_id: integer (nullable = true)
         |-- name: string (nullable = true)
         |-- age: long (nullable = true)
         |-- skills!.name: string (nullable = true)
         |-- skills!.level: string (nullable = true)
         |-- projects!.name: string (nullable = true)
         |-- projects!.client: string (nullable = true)
         |-- projects!.tasks!.name: string (nullable = true)
         |-- projects!.tasks!.description: string (nullable = true)
         |-- projects!.tasks!.status: string (nullable = true)
         |-- projects!.tasks!.estimate: double (nullable = true)
        <BLANKLINE>
        >>> new_df.select("employee_id", "name", "age", "skills").show(truncate=False)
        +-----------+----------+---+---------------------------------------------+
        |employee_id|name      |age|skills                                       |
        +-----------+----------+---+---------------------------------------------+
        |1          |John Smith|30 |[{Java, EXPERT}, {Python, INTERMEDIATE}]     |
        |2          |Jane Doe  |25 |[{JavaScript, ADVANCED}, {PHP, INTERMEDIATE}]|
        +-----------+----------+---+---------------------------------------------+
        <BLANKLINE>

        As we can see, we obtained the same result with a much simpler and cleaner code.
        Now let's explain what this code did:

        The [`spark_frame.nested.with_fields`]
        [spark_frame.nested_impl.with_fields.with_fields] method is similar to
        the [`pyspark.sql.DataFrame.withColumns`][] method, except that it works on nested fields inside
        structs and arrays. We pass it a `Dict(field_name, transformation)`
        indicating the expression we want to apply for each field. The transformation must be a higher order function:
        a lambda expression or named function that takes a Column as argument and returns a Column. The column passed
        to that function will represent the struct parent of the target field. For instance, when we write
        `"skills!.level": lambda skill: f.upper(skill["level"])`, the lambda function will be applied to each struct
        element of the array `skills`.

        !!! Info
            _The data for this example was generated by ChatGPT :-)_
    """
    # This is a hacky way to have doctests that runs in the pipeline and are usable in the doc thanks to mkdocstrings


def select_nested_fields():
    """
    Examples: In this example, we will see how to select and rename specific elements in a nested data structure

        >>> from spark_frame.examples.working_with_nested_data import _get_sample_employee_data
        >>> from pyspark.sql import functions as f
        >>> from spark_frame import nested
        >>> df = _get_sample_employee_data()
        >>> nested.print_schema(df)
        root
         |-- employee_id: integer (nullable = true)
         |-- name: string (nullable = true)
         |-- age: long (nullable = true)
         |-- skills!.name: string (nullable = true)
         |-- skills!.level: string (nullable = true)
         |-- projects!.name: string (nullable = true)
         |-- projects!.client: string (nullable = true)
         |-- projects!.tasks!.name: string (nullable = true)
         |-- projects!.tasks!.description: string (nullable = true)
         |-- projects!.tasks!.status: string (nullable = true)
         |-- projects!.tasks!.estimate: long (nullable = true)
        <BLANKLINE>
        >>> df.show(truncate=False)  # noqa: E501
        +-----------+----------+---+---------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |employee_id|name      |age|skills                                       |projects                                                                                                                                                                                                                          |
        +-----------+----------+---+---------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |1          |John Smith|30 |[{Java, expert}, {Python, intermediate}]     |[{Project A, Acme Inc, [{Task 1, Implement feature X, completed, 8}, {Task 2, Fix bug Y, in progress, 5}]}, {Project B, Beta Corp, [{Task 3, Implement feature Z, pending, 13}, {Task 4, Improve performance, in progress, 3}]}]  |
        |2          |Jane Doe  |25 |[{JavaScript, advanced}, {PHP, intermediate}]|[{Project C, Gamma Inc, [{Task 5, Implement feature W, completed, 20}, {Task 6, Fix bug V, in progress, 13}]}, {Project D, Delta Ltd, [{Task 7, Implement feature U, pending, 8}, {Task 8, Improve performance, in progress, 5}]}]|
        +-----------+----------+---+---------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        <BLANKLINE>

        ### The task
        Let's say we want to select only the following fields, while keeping the same overall structure:
        - employee_id
        - projects.name
        - projects.tasks.name

        ### Without spark_frame.nested
        This forces us to do something quite complicated, using [pyspark.sql.functions.transform][]
        >>> new_df = df.select(
        ...     "employee_id",
        ...     f.transform("projects", lambda project:
        ...         f.struct(project["name"].alias("name"), f.transform(project["tasks"], lambda task:
        ...             f.struct(task["name"].alias("name"))
        ...         ).alias("tasks"))
        ...     ).alias("projects")
        ... )
        >>> nested.print_schema(new_df)
        root
         |-- employee_id: integer (nullable = true)
         |-- projects!.name: string (nullable = true)
         |-- projects!.tasks!.name: string (nullable = true)
        <BLANKLINE>
        >>> new_df.show(truncate=False)
        +-----------+----------------------------------------------------------------------+
        |employee_id|projects                                                              |
        +-----------+----------------------------------------------------------------------+
        |1          |[{Project A, [{Task 1}, {Task 2}]}, {Project B, [{Task 3}, {Task 4}]}]|
        |2          |[{Project C, [{Task 5}, {Task 6}]}, {Project D, [{Task 7}, {Task 8}]}]|
        +-----------+----------------------------------------------------------------------+
        <BLANKLINE>

        ### With spark_frame.nested
        Using [`spark_frame.nested.select`][spark_frame.nested_impl.select_impl.select], we can easily obtain the exact
        same result.
        >>> new_df = df.transform(nested.select, {
        ...     "employee_id": None,
        ...     "projects!.name": None,
        ...     "projects!.tasks!.name": None
        ... })
        >>> nested.print_schema(new_df)
        root
         |-- employee_id: integer (nullable = true)
         |-- projects!.name: string (nullable = true)
         |-- projects!.tasks!.name: string (nullable = true)
        <BLANKLINE>
        >>> new_df.show(truncate=False)
        +-----------+----------------------------------------------------------------------+
        |employee_id|projects                                                              |
        +-----------+----------------------------------------------------------------------+
        |1          |[{Project A, [{Task 1}, {Task 2}]}, {Project B, [{Task 3}, {Task 4}]}]|
        |2          |[{Project C, [{Task 5}, {Task 6}]}, {Project D, [{Task 7}, {Task 8}]}]|
        +-----------+----------------------------------------------------------------------+
        <BLANKLINE>

        Here, `None` is used to indicate that we don't want to perform any transformation on the column, be we could
        also replace them with functions to perform transformations at the same time. For instance, we could pass
        all the names to uppercase like this:
        >>> df.transform(nested.select, {
        ...     "employee_id": None,
        ...     "projects!.name": lambda project: f.upper(project["name"]),
        ...     "projects!.tasks!.name": lambda task: f.upper(task["name"])
        ... }).show(truncate=False)
        +-----------+----------------------------------------------------------------------+
        |employee_id|projects                                                              |
        +-----------+----------------------------------------------------------------------+
        |1          |[{PROJECT A, [{TASK 1}, {TASK 2}]}, {PROJECT B, [{TASK 3}, {TASK 4}]}]|
        |2          |[{PROJECT C, [{TASK 5}, {TASK 6}]}, {PROJECT D, [{TASK 7}, {TASK 8}]}]|
        +-----------+----------------------------------------------------------------------+
        <BLANKLINE>
    """
    # This is a hacky way to have doctests that runs in the pipeline and are usable in the doc thanks to mkdocstrings


def _get_sample_employee_data():
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
          "skills": [
            {
              "name": "Java",
              "level": "expert"
            },
            {
              "name": "Python",
              "level": "intermediate"
            }
          ],
          "projects": [
            {
              "name": "Project A",
              "client": "Acme Inc",
              "tasks": [
                {
                  "name": "Task 1",
                  "description": "Implement feature X",
                  "status": "completed",
                  "estimate": 8
                },
                {
                  "name": "Task 2",
                  "description": "Fix bug Y",
                  "status": "in progress",
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
                  "description": "Implement feature Z",
                  "status": "pending",
                  "estimate": 13
                },
                {
                  "name": "Task 4",
                  "description": "Improve performance",
                  "status": "in progress",
                  "estimate": 3
                }
              ]
            }
          ]
        },
        {
          "employee_id": 2,
          "name": "Jane Doe",
          "age": 25,
          "skills": [
            {
              "name": "JavaScript",
              "level": "advanced"
            },
            {
              "name": "PHP",
              "level": "intermediate"
            }
          ],
          "projects": [
            {
              "name": "Project C",
              "client": "Gamma Inc",
              "tasks": [
                {
                  "name": "Task 5",
                  "description": "Implement feature W",
                  "status": "completed",
                  "estimate": 20
                },
                {
                  "name": "Task 6",
                  "description": "Fix bug V",
                  "status": "in progress",
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
                  "description": "Implement feature U",
                  "status": "pending",
                  "estimate": 8
                },
                {
                  "name": "Task 8",
                  "description": "Improve performance",
                  "status": "in progress",
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
                  "name": "skills",
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
                          "name": "level",
                          "type": "string",
                          "nullable": true,
                          "metadata": {}
                        }
                      ]
                    }
                  },
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
                                  "name": "description",
                                  "type": "string",
                                  "nullable": true,
                                  "metadata": {}
                                },
                                {
                                  "name": "status",
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
