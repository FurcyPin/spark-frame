from typing import Callable, Dict, List, Optional, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import StructField

from spark_frame import nested
from spark_frame.nested_impl.unnest_all_fields import unnest_all_fields
from spark_frame.transformations_impl import analyze_aggs
from spark_frame.transformations_impl.union_dataframes import union_dataframes
from spark_frame.utils import is_sub_field_of_any, quote


def _analyze_flat_df(
    flat_df: DataFrame,
    index_by_field: Dict[str, int],
    group_by: List[str],
    aggs: List[Callable[[str, StructField, int], Column]],
):
    agg_alias = "_agg"

    def agg_struct(col: StructField):
        """Builds a struct will all aggregation function results for the given column"""
        return f.struct([agg(quote(col.name), col, index_by_field[col.name]) for agg in aggs])

    aggregation_per_field = [agg_struct(col) for col in flat_df.schema.fields if col.name not in group_by]
    aggregation_for_all_fields = f.array(aggregation_per_field).alias(agg_alias)
    if len(group_by) > 0:
        res = flat_df.groupby(group_by).agg(aggregation_for_all_fields)
    else:
        res = flat_df.select(aggregation_for_all_fields)
    res = res.select(*group_by, f.explode(agg_alias).alias(agg_alias))
    res = res.select(*group_by, agg_alias + ".*")
    return res


default_aggs: List[Callable[[str, StructField, int], Column]] = [
    analyze_aggs.column_number,
    analyze_aggs.column_name,
    analyze_aggs.column_type,
    analyze_aggs.count,
    analyze_aggs.count_distinct,
    analyze_aggs.count_null,
    analyze_aggs.min,
    analyze_aggs.max,
]


def analyze(
    df: DataFrame,
    group_by: Optional[Union[str, List[str]]] = None,
    group_alias: str = "group",
    _aggs: Optional[List[Callable[[str, StructField, int], Column]]] = None,
) -> DataFrame:
    """Analyze a DataFrame by computing various stats for each column.

    By default, it returns a DataFrame with one row per column and the following columns
    (but the columns computed can be customized, see the Customization section below):

    - `column_number`: Number of the column (useful for sorting)
    - `column_name`: Name of the column
    - `column_type`: Type of the column
    - `count`: Number of rows in the column, it is equal to the number of rows in the table, except for columns nested
      `inside` arrays for which it may be different
    - `count_distinct`: Number of distinct values
    - `count_null`: Number of null values
    - `min`: smallest value
    - `max`: largest value

    Implementation details
    ----------------------
    - Structs are flattened with a `.` after their name.
    - Arrays are unnested with a `!` character after their name, which is why they may have a different count.
    - Null values are not counted in the count_distinct column.

    !!! warning "Limitation: Map type is not supported"
        This method currently does not work on columns of type Map.
        A possible workaround is to use [`spark_frame.transformations.convert_all_maps_to_arrays`]
        [spark_frame.transformations_impl.convert_all_maps_to_arrays.convert_all_maps_to_arrays]
        before using it.

    Grouping
    --------
    With the `group_by` option, users can specify one or multiple columns for which the statistics will be grouped.
    If this option is used, an extra column "group" of type struct will be added to output DataFrame.
    See the examples below.

    !!! warning "Limitation: group_by only works on non-repeated fields"
        Currently, the `group_by` option only works with non-repeated fields.
        Using it on repeated fields will lead to an unspecified error.

    Customization
    -------------
    By default, this method will compute for each column the aggregations listed in
    `spark_frame.transformation_impl.analyze.default_aggs`, but users can change this and even add their
    own custom aggregation by passing the argument `_agg`, a list of aggregation functions with the following
    signature: `(col: str, schema_field: StructField, col_num: int) -> Column`

    Examples of aggregation methods can be found in the module `spark_frame.transformation_impl.analyze_aggs`

    Args:
        df: A Spark DataFrame
        group_by: A list of column names on which the aggregations will be grouped
        group_alias: The alias to use for the struct column that will contain the `group_by` columns, if any.
        _aggs: A list of aggregation to override the default aggregation made by the function

    Returns:
        A new DataFrame containing descriptive statistics about the input DataFrame

    Examples:
        >>> from spark_frame.transformations_impl.analyze import __get_test_df

        >>> df = __get_test_df()
        >>> df.show()
        +---+----------+---------------+------------+
        | id|      name|          types|   evolution|
        +---+----------+---------------+------------+
        |  1| Bulbasaur|[Grass, Poison]|{true, null}|
        |  2|   Ivysaur|[Grass, Poison]|   {true, 1}|
        |  3|  Venusaur|[Grass, Poison]|  {false, 2}|
        |  4|Charmander|         [Fire]|{true, null}|
        |  5|Charmeleon|         [Fire]|   {true, 4}|
        |  6| Charizard| [Fire, Flying]|  {false, 5}|
        |  7|  Squirtle|        [Water]|{true, null}|
        |  8| Wartortle|        [Water]|   {true, 7}|
        |  9| Blastoise|        [Water]|  {false, 8}|
        +---+----------+---------------+------------+
        <BLANKLINE>

        >>> analyzed_df = analyze(df)
        Analyzing 5 columns ...
        >>> analyzed_df.show(truncate=False)  # noqa: E501
        +-------------+----------------------+-----------+-----+--------------+----------+---------+---------+
        |column_number|column_name           |column_type|count|count_distinct|count_null|min      |max      |
        +-------------+----------------------+-----------+-----+--------------+----------+---------+---------+
        |0            |id                    |INTEGER    |9    |9             |0         |1        |9        |
        |1            |name                  |STRING     |9    |9             |0         |Blastoise|Wartortle|
        |2            |types!                |STRING     |13   |5             |0         |Fire     |Water    |
        |3            |evolution.can_evolve  |BOOLEAN    |9    |2             |0         |false    |true     |
        |4            |evolution.evolves_from|INTEGER    |9    |6             |3         |1        |8        |
        +-------------+----------------------+-----------+-----+--------------+----------+---------+---------+
        <BLANKLINE>

        >>> df = __get_test_df().withColumn("main_type", f.expr("types[0]"))
        >>> df.show()
        +---+----------+---------------+------------+---------+
        | id|      name|          types|   evolution|main_type|
        +---+----------+---------------+------------+---------+
        |  1| Bulbasaur|[Grass, Poison]|{true, null}|    Grass|
        |  2|   Ivysaur|[Grass, Poison]|   {true, 1}|    Grass|
        |  3|  Venusaur|[Grass, Poison]|  {false, 2}|    Grass|
        |  4|Charmander|         [Fire]|{true, null}|     Fire|
        |  5|Charmeleon|         [Fire]|   {true, 4}|     Fire|
        |  6| Charizard| [Fire, Flying]|  {false, 5}|     Fire|
        |  7|  Squirtle|        [Water]|{true, null}|    Water|
        |  8| Wartortle|        [Water]|   {true, 7}|    Water|
        |  9| Blastoise|        [Water]|  {false, 8}|    Water|
        +---+----------+---------------+------------+---------+
        <BLANKLINE>

        >>> from spark_frame.transformations_impl import analyze_aggs
        >>> aggs = [
        ...     analyze_aggs.column_number,
        ...     analyze_aggs.column_name,
        ...     analyze_aggs.count,
        ...     analyze_aggs.count_distinct,
        ...     analyze_aggs.count_null,
        ... ]
        >>> analyzed_df = analyze(df, group_by="main_type", _aggs=aggs)
        Analyzing 5 columns ...
        >>> analyzed_df.orderBy("`group`.main_type", "column_number").show(truncate=False)
        +-------+-------------+----------------------+-----+--------------+----------+
        |group  |column_number|column_name           |count|count_distinct|count_null|
        +-------+-------------+----------------------+-----+--------------+----------+
        |{Fire} |0            |id                    |3    |3             |0         |
        |{Fire} |1            |name                  |3    |3             |0         |
        |{Fire} |2            |types!                |4    |2             |0         |
        |{Fire} |3            |evolution.can_evolve  |3    |2             |0         |
        |{Fire} |4            |evolution.evolves_from|3    |2             |1         |
        |{Grass}|0            |id                    |3    |3             |0         |
        |{Grass}|1            |name                  |3    |3             |0         |
        |{Grass}|2            |types!                |6    |2             |0         |
        |{Grass}|3            |evolution.can_evolve  |3    |2             |0         |
        |{Grass}|4            |evolution.evolves_from|3    |2             |1         |
        |{Water}|0            |id                    |3    |3             |0         |
        |{Water}|1            |name                  |3    |3             |0         |
        |{Water}|2            |types!                |3    |1             |0         |
        |{Water}|3            |evolution.can_evolve  |3    |2             |0         |
        |{Water}|4            |evolution.evolves_from|3    |2             |1         |
        +-------+-------------+----------------------+-----+--------------+----------+
        <BLANKLINE>
    """
    if _aggs is None:
        _aggs = default_aggs
    if group_by is None:
        group_by = []
    if isinstance(group_by, str):
        group_by = [group_by]

    flat_fields = nested.fields(df)
    fields_to_drop = [field for field in flat_fields if is_sub_field_of_any(field, group_by)]
    nb_cols = len(flat_fields) - len(fields_to_drop)
    print(f"Analyzing {nb_cols} columns ...")

    if len(group_by) > 0:
        df = df.withColumn(group_alias, f.struct(*group_by))
        group = [group_alias]
    else:
        group = []

    flattened_dfs = unnest_all_fields(df, keep_columns=group)
    index_by_field = {field: index for index, field in enumerate(flat_fields)}
    analyzed_dfs = [
        _analyze_flat_df(flat_df.drop(*fields_to_drop), index_by_field, group_by=group, aggs=_aggs)
        for flat_df in flattened_dfs
    ]

    union_df = union_dataframes(*analyzed_dfs)
    return union_df.orderBy("column_number")


def __get_test_df() -> DataFrame:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("doctest").getOrCreate()
    query = """
        SELECT INLINE(ARRAY(
            STRUCT(
                1 as id, "Bulbasaur" as name, ARRAY("Grass", "Poison") as types,
                STRUCT(TRUE as can_evolve, NULL as evolves_from) as evolution
            ),
            STRUCT(
                2 as id, "Ivysaur" as name, ARRAY("Grass", "Poison") as types,
                STRUCT(TRUE as can_evolve, 1 as evolves_from) as evolution
            ),
            STRUCT(
                3 as id, "Venusaur" as name, ARRAY("Grass", "Poison") as types,
                STRUCT(FALSE as can_evolve, 2 as evolves_from) as evolution
            ),
            STRUCT(
                4 as id, "Charmander" as name, ARRAY("Fire") as types,
                STRUCT(TRUE as can_evolve, NULL as evolves_from) as evolution
            ),
            STRUCT(
                5 as id, "Charmeleon" as name, ARRAY("Fire") as types,
                STRUCT(TRUE as can_evolve, 4 as evolves_from) as evolution
            ),
            STRUCT(
                6 as id, "Charizard" as name, ARRAY("Fire", "Flying") as types,
                STRUCT(FALSE as can_evolve, 5 as evolves_from) as evolution
            ),
            STRUCT(
                7 as id, "Squirtle" as name, ARRAY("Water") as types,
                STRUCT(TRUE as can_evolve, NULL as evolves_from) as evolution
            ),
            STRUCT(
                8 as id, "Wartortle" as name, ARRAY("Water") as types,
                STRUCT(TRUE as can_evolve, 7 as evolves_from) as evolution
            ),
            STRUCT(
                9 as id, "Blastoise" as name, ARRAY("Water") as types,
                STRUCT(FALSE as can_evolve, 8 as evolves_from) as evolution
            )
        ))
    """
    return spark.sql(query)
