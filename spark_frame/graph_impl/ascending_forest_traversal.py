from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from spark_frame.utils import assert_true, quote


def ascending_forest_traversal(df: DataFrame, id: str, parent_id: str) -> DataFrame:
    """Given a DataFrame representing a labeled forest with columns "id", "parent_id" and other label columns,
    performs a graph traversal that will return a DataFrame with the same schema that gives for each node
    the labels of it's furthest ancestor.

    This algorithm is optimized for lowly-connected graphs that fit in RAM.
    In other words, for a graph G = (V, E) we assume that |E| << |V|

    It has a security against dependency cycles, but no security preventing
    a combinatorics explosion if some nodes have more than one parent.

    Example:

    >>> from pyspark.sql import SparkSession
    >>> spark = SparkSession.builder.appName("doctest").getOrCreate()

    # Given a DataFrame with pokemon attributes and evolution links

    >>> df = spark.sql('''
    ...     SELECT
    ...       col1 as `pokemon.id`,
    ...       col2 as `pokemon.evolve_to_id`,
    ...       col3 as `pokemon.name`,
    ...       col4 as `pokemon.types`
    ...     FROM VALUES
    ...       (4, 5, 'Charmander', ARRAY('Fire')),
    ...       (5, 6, 'Charmeleon', ARRAY('Fire')),
    ...       (6, NULL, 'Charizard', ARRAY('Fire', 'Flying'))
    ... ''')
    >>> df.show()
    +----------+--------------------+------------+--------------+
    |pokemon.id|pokemon.evolve_to_id|pokemon.name| pokemon.types|
    +----------+--------------------+------------+--------------+
    |         4|                   5|  Charmander|        [Fire]|
    |         5|                   6|  Charmeleon|        [Fire]|
    |         6|                null|   Charizard|[Fire, Flying]|
    +----------+--------------------+------------+--------------+
    <BLANKLINE>

    # We compute a DataFrame that for each pokemon.id gives the attributes of its highest level of evolution

    >>> ascending_forest_traversal(df, "pokemon.id", "pokemon.evolve_to_id").orderBy("`pokemon.id`").show()
    +----------+--------------------+------------+--------------+
    |pokemon.id|pokemon.evolve_to_id|pokemon.name| pokemon.types|
    +----------+--------------------+------------+--------------+
    |         4|                null|   Charizard|[Fire, Flying]|
    |         5|                null|   Charizard|[Fire, Flying]|
    |         6|                null|   Charizard|[Fire, Flying]|
    +----------+--------------------+------------+--------------+
    <BLANKLINE>

    :param df: a Spark DataFrame
    :param id: name of the column that represent the node's ids
    :param parent_id: name of the column that represent the parent node's ids
    :return: a DataFrame with the same schema as the input DataFrame that gives
        for each node the labels of it's furthest ancestor
    """
    assert_true(id in df.columns, "Could not find column %s in dataframe's columns: %s" % (id, df.columns))
    assert_true(
        parent_id in df.columns, "Could not find column %s in dataframe's columns: %s" % (parent_id, df.columns)
    )
    df = df.repartition(200, f.col(quote(id))).persist()
    df_null = df.where(f.col(quote(parent_id)).isNull())
    df_not_null = df.where(f.col(quote(parent_id)).isNotNull()).persist()
    do_continue = True

    while do_continue:
        joined_df_not_null = (
            df_not_null.alias("a")
            .join(df.alias("b"), f.col("a." + quote(parent_id)) == f.col("b." + quote(id)), "left")
            .select(
                f.col("a." + quote(id)).alias(id),
                f.when(f.col("b." + quote(parent_id)) == f.col("a." + quote(id)), f.lit(None))
                .otherwise(f.col("b." + quote(parent_id)))
                .alias(parent_id),
                *[
                    f.coalesce(f.col("b." + quote(col)), f.col("a." + quote(col))).alias(col)
                    for col in df.columns
                    if col not in (id, parent_id)
                ],
            )
        )
        joined_df_not_null = joined_df_not_null.persist()
        new_df_not_null = joined_df_not_null.where(f.col(quote(parent_id)).isNotNull()).persist()
        do_continue = new_df_not_null.count() > 0
        new_df_null = df_null.union(joined_df_not_null.where(f.col(quote(parent_id)).isNull()))

        df_not_null = new_df_not_null
        df_null = new_df_null

    return df_null.union(df_not_null)
