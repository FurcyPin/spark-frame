from pyspark.sql import SparkSession

from spark_frame.transformations import parse_json_columns


def _get_sample_pokemon_data():
    spark = SparkSession.builder.appName("doctest").getOrCreate()
    json_str = """
        {
            "id": 1,
            "name": {
              "english": "Bulbasaur",
              "french": "Bulbizarre"
            },
            "types": [
              "Grass",
              "Poison"
            ],
            "base_stats": {
              "HP": 45,
              "Attack": 49,
              "Defense": 49,
              "Sp Attack": 65,
              "Sp Defense": 65,
              "Speed": 45
            }
          }
    """.replace(
        "\n", ""
    )
    df = spark.createDataFrame([(json_str,)], "value STRING").repartition(1)
    df = parse_json_columns(df, "value").select("value.*")
    return df


def transform_nested_fields():
    """
    This example demonstrates how the [flatten](/reference/#spark_frame.transformations_impl.flatten.flatten) and
    unflatten [unflatten](/reference/#spark_frame.transformations_impl.unflatten.unflatten) methods can be used to
    make data cleaning pipeline easier with PySpark.

    Examples: Let's take a sample DataFrame with our favorite example: Pokemons!

        >>> from spark_frame.examples.working_with_nested_data import _get_sample_pokemon_data
        >>> df = _get_sample_pokemon_data()
        >>> df.printSchema()
        root
         |-- base_stats: struct (nullable = true)
         |    |-- Attack: long (nullable = true)
         |    |-- Defense: long (nullable = true)
         |    |-- HP: long (nullable = true)
         |    |-- Sp Attack: long (nullable = true)
         |    |-- Sp Defense: long (nullable = true)
         |    |-- Speed: long (nullable = true)
         |-- id: long (nullable = true)
         |-- name: struct (nullable = true)
         |    |-- english: string (nullable = true)
         |    |-- french: string (nullable = true)
         |-- types: array (nullable = true)
         |    |-- element: string (containsNull = true)
        <BLANKLINE>
        >>> df.show(vertical=True, truncate=False)  # doctest: +NORMALIZE_WHITESPACE
        -RECORD 0------------------------------
         base_stats | {49, 49, 45, 65, 65, 45}
         id         | 1
         name       | {Bulbasaur, Bulbizarre}
         types      | [Grass, Poison]
        <BLANKLINE>

        Let's say we want to add a new enrich the "base_stats" struct with a new field named "Total".

        ### Without spark-frame
        Of course, we could write something in DataFrame or SQL like this:

        >>> df.createOrReplaceTempView("df")
        >>> new_df = df.sparkSession.sql('''
        ... SELECT
        ...   STRUCT(
        ...     base_stats.*,
        ...     base_stats.Attack + base_stats.Defense + base_stats.HP +
        ...     base_stats.`Sp Attack` + base_stats.`Sp Defense` + base_stats.Speed as Total
        ...   ) as base_stats,
        ...   id,
        ...   name,
        ...   types
        ... FROM df
        ... ''').show(vertical=True, truncate=False)  # doctest: +NORMALIZE_WHITESPACE
        -RECORD 0-----------------------------------
         base_stats | {49, 49, 45, 65, 65, 45, 318}
         id         | 1
         name       | {Bulbasaur, Bulbizarre}
         types      | [Grass, Poison]
        <BLANKLINE>

        It works, but it is a little cumbersome. Imagine how ugly the query would look like with a much bigger table,
        with hundreds of columns with three levels of nesting or more...

        ### With spark-frame
        Instead, we can use the [flatten](/reference/#spark_frame.transformations_impl.flatten.flatten) and
        unflatten [unflatten](/reference/#spark_frame.transformations_impl.unflatten.unflatten) method to reduce
        boilerplate significantly.

        >>> from spark_frame.transformations import flatten, unflatten
        >>> from pyspark.sql import functions as f
        >>> flat_df = flatten(df)
        >>> flat_df = flat_df.withColumn("base_stats.Total",
        ...     f.col("`base_stats.Attack`") + f.col("`base_stats.Defense`") + f.col("`base_stats.HP`") +
        ...     f.col("`base_stats.Sp Attack`") + f.col("`base_stats.Sp Defense`") + f.col("`base_stats.Speed`")
        ... )
        >>> new_df = unflatten(flat_df)
        >>> new_df.show(vertical=True, truncate=False)  # doctest: +NORMALIZE_WHITESPACE
        -RECORD 0-----------------------------------
         base_stats | {49, 49, 45, 65, 65, 45, 318}
         id         | 1
         name       | {Bulbasaur, Bulbizarre}
         types      | [Grass, Poison]
        <BLANKLINE>

        This yield the same result, and we did not have to mention the names of the columns we did not care about.
        This makes pipelines much easier to maintain. If a new column is added to your source table, you don't need
        to update this data enrichment code to propagate it automatically. On the other hand, with the first SQL
        solution, you would have had to specifically add this new field to the query to propagate it.

        We can even use [DataFrame.transform](pyspark.sql.DataFrame.transform) to inline everything!

        >>> df.transform(flatten).withColumn(
        ...     "base_stats.Total",
        ...     f.col("`base_stats.Attack`") + f.col("`base_stats.Defense`") + f.col("`base_stats.HP`") +
        ...     f.col("`base_stats.Sp Attack`") + f.col("`base_stats.Sp Defense`") + f.col("`base_stats.Speed`")
        ...   ).transform(unflatten).show(vertical=True, truncate=False)  # doctest: +NORMALIZE_WHITESPACE
        -RECORD 0-----------------------------------
         base_stats | {49, 49, 45, 65, 65, 45, 318}
         id         | 1
         name       | {Bulbasaur, Bulbizarre}
         types      | [Grass, Poison]
        <BLANKLINE>

        !!! Info
            _This example uses data taken from
            [https://raw.githubusercontent.com/fanzeyi/pokemon.json/master/pokedex.json](
            https://raw.githubusercontent.com/fanzeyi/pokemon.json/master/pokedex.json).
    """
    # This is a hacky way to have doctests that runs in the pipeline and are usable in the doc thanks to mkdocstrings
