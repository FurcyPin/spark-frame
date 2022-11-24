from typing import Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from spark_frame.utils import assert_true, quote


def _ascending_forest_traversal(
    input_df: DataFrame,
    node_id_col_name: str = "node_id",
    parent_id_col_name: str = "parent_id",
    highest_parent_id_col_name: str = "highest_parent_id",
    status_col_name: str = "status",
    incomplete_status: Union[str, int] = 0,
    done_status: Union[str, int] = 1,
    cycle_status: Union[str, int] = -1,
):
    """Perform an ascending forest traversal on a DataFrame with fixed column names.
    Columns names may be customized using extra parameters.

    The following constraints are mandatory and must be ensured beforehand by the user, otherwise the algorithm
    might go wrong.

    - Schema: The input DataFrame must have two columns "node_id" and "parent_id", which must have compatible and
        sortable types. Int, bigint or string are recommended. Column names may be customized with `id_col_name`
        and `parent_id_col_name`.
    - Unicity: the input DataFrame must have at most one row per "node_id". But distinct "id"s may have the
        same "parent_id".

    The algorithm works as follows:

    Algorithm:

        Initialization:
            - All nodes that have no parent are marked as done (`status="done"`)
        Algorithm loop:
            1. Each node looks at its parent and replaces its parent with its parent's parent.
            2. If its parent is "done", it is marked as "done" too.
            3. Go back to 1. until all nodes are "done".

    Assuming the graph contains no cycle, the algorithm terminates in `O(log(n))` because the distance between 2 nodes
    is divided by 2 at every loop iteration. More precisely, if at any iteration there is a path of size `2*k` or
    `2*k+1` between a node and its furthest ancestor, then at the next iteration a path of size `k` will exist.

    However, the above algorithm may loop indefinitely if the graph, contains a cycle, which is why we modify the
    algorithm in the following way to have a "cycle detection security".

    Algorithm:

        Initialization:
            - All nodes that have no parent are marked as done (`status="done"`)
            - *A column "highest_parent_id" is added that initially contains the "parent_id"*
        Algorithm loop:
            1. Each node looks at its parent and replaces its parent with its parent's parent,
            2. *If a node and their parent are not "done" and have the same "highest_parent_id", then a cycle has been
                detected. (It means that the node had already 'known' about this highest parent _before_ the information
                from its parent was propagated, which can only happen if there is a cycle).
                In that case, the node's status is marked as "cycle". Otherwise, their
                "highest_parent_id" becomes the highest value between theirs and their parents*.
            3. If its parent is "done", it is marked as "done" too. *If its parent is "cycle", it is marked as "cycle"
                too.*
            4. Go back to 1. until all nodes are "done".

    This algorithm terminates in `O(log(n))` if the graph contains no loop and `O(2 log(n))` if there is a cycle.
    This is because if there is a cycle, all nodes in the cycle will eventually get the highest id (let's call it _`H`_)
    in the cycle as their "highest_parent_id", which will take `O(log(n))` iterations. Indeed, if at any step, there is
    a path of length `2*k` or `2*k+1` between any node and "a node which has _`highest_parent_id == H`_", then at the
    next iteration, there will exist a path of length `k` between them this node and "a node which has
    _`highest_parent_id == H`_".
    The case `2*k` is clear as you can easily build a path between the two nodes, but the case `2*k+1` is a little more
    tricky: you can build a path between the first and the `2*k`-th node and prove that this node's
    "highest_parent_id" will become equal to _`H`_ at the next iteration.
    Similarly, once the cycle is detected, the information will take `O(log(n))` to reach all nodes in the cycle.


    Args:
        input_df: a DataFrame with 2 columns: "node_id", "parent_id"
        node_id_col_name:
        parent_id_col_name:
        status_col_name:
        incomplete_status:
        done_status:
        cycle_status:

    Returns:
        a DataFrame

    Test: Simple case

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()

        >>> input_df = spark.sql('''
        ...     SELECT
        ...       col1 as `node_id`,
        ...       col2 as `parent_id`
        ...     FROM VALUES (1, 2), (2, 3), (3, 3)
        ... ''')
        >>> input_df.show()
        +-------+---------+
        |node_id|parent_id|
        +-------+---------+
        |      1|        2|
        |      2|        3|
        |      3|        3|
        +-------+---------+
        <BLANKLINE>
        >>> _ascending_forest_traversal(input_df).orderBy("node_id").show()
        +-------+---------+------+
        |node_id|parent_id|status|
        +-------+---------+------+
        |      1|        3|     1|
        |      2|        3|     1|
        |      3|        3|     1|
        +-------+---------+------+
        <BLANKLINE>

    Test: Cycle case

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()

        >>> input_df = spark.sql('''
        ...     SELECT
        ...       col1 as `node_id`,
        ...       col2 as `parent_id`
        ...     FROM VALUES (1, 2), (2, 3), (3, 1)
        ... ''')
        >>> input_df.show()
        +-------+---------+
        |node_id|parent_id|
        +-------+---------+
        |      1|        2|
        |      2|        3|
        |      3|        1|
        +-------+---------+
        <BLANKLINE>
        >>> _ascending_forest_traversal(input_df).orderBy("node_id").show()
        +-------+---------+------+
        |node_id|parent_id|status|
        +-------+---------+------+
        |      1|     null|    -1|
        |      2|     null|    -1|
        |      3|     null|    -1|
        +-------+---------+------+
        <BLANKLINE>

    Test: Case with parent that does not exist

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()

        >>> input_df = spark.sql('''
        ...     SELECT
        ...       col1 as `node_id`,
        ...       col2 as `parent_id`
        ...     FROM VALUES (1, 2), (2, 3), (3, 4)
        ... ''')
        >>> input_df.show()
        +-------+---------+
        |node_id|parent_id|
        +-------+---------+
        |      1|        2|
        |      2|        3|
        |      3|        4|
        +-------+---------+
        <BLANKLINE>
        >>> _ascending_forest_traversal(input_df).orderBy("node_id").show()
        +-------+---------+------+
        |node_id|parent_id|status|
        +-------+---------+------+
        |      1|        4|     1|
        |      2|        4|     1|
        |      3|        4|     1|
        +-------+---------+------+
        <BLANKLINE>
    """
    # node_id_col_name = "node_id"
    # parent_id_col_name = "parent_id"
    # highest_parent_id_col_name = "highest_parent_id"
    # status_col_name = "status"
    # incomplete_status = 0
    # done_status = 1
    # cycle_status = -1
    # input_df = spark.sql('''
    #     SELECT
    #       col1 as `node_id`,
    #       col2 as `parent_id`
    #     FROM VALUES (1, 2), (2, 3), (3, 3), (11, 12), (12, 13), (13, 11)
    # ''')
    df = input_df
    # df.show()
    # +-------+---------+
    # |node_id|parent_id|
    # +-------+---------+
    # |      1|        2|
    # |      2|        3|
    # |      3|        3|
    # |     11|       12|
    # |     12|       13|
    # |     13|       11|
    # +-------+---------+
    node_id_col = f.col(node_id_col_name)
    parent_id_col = f.col(parent_id_col_name)
    status_col = f.col(status_col_name)
    done_status_col = f.lit(done_status)
    incomplete_status_col = f.lit(incomplete_status)
    cycle_status_col = f.lit(cycle_status)

    # Initialization:
    df = df.select(node_id_col_name, f.coalesce(parent_id_col, node_id_col).alias(parent_id_col_name))
    # - A column "highest_parent_id" is added that initially contains the "parent_id"
    df = df.withColumn(highest_parent_id_col_name, parent_id_col)
    # - All nodes that have no parent are marked as done (`status="done"`)
    df = df.withColumn(
        status_col_name,
        f.when(node_id_col == parent_id_col, done_status_col).otherwise(incomplete_status_col).alias(status_col_name),
    )
    # df.show()
    # +-------+---------+-----------------+------+
    # |node_id|parent_id|highest_parent_id|status|
    # +-------+---------+-----------------+------+
    # |      1|        2|                2|     0|
    # |      2|        3|                3|     0|
    # |      3|        3|                3|     1|
    # |     11|       12|               12|     0|
    # |     12|       13|               13|     0|
    # |     13|       11|               11|     0|
    # +-------+---------+-----------------+------+

    df_done = df.where(status_col.isin([done_status_col, cycle_status_col]))
    # df_done.show()
    df_incomplete = df.where(~status_col.isin([done_status_col, cycle_status_col])).persist()
    # df_incomplete.show()

    # Algorithm loop:
    do_continue = True
    while do_continue:
        joined_df = df_incomplete.alias("a").join(
            df_incomplete.union(df_done).alias("b"),
            f.col("a." + parent_id_col_name) == f.col("b." + node_id_col_name),
            "left",
        )
        new_node_id_col = f.col("a." + node_id_col_name).alias(node_id_col_name)
        # 1. Each node replaces its parent with its parent's parent,
        new_parent_id_col = f.coalesce(f.col("b." + parent_id_col_name), f.col("a." + parent_id_col_name)).alias(
            parent_id_col_name
        )

        # 2. If a node and their parent are not "done" and have the same "highest_parent_id", then a cycle has been
        #    detected. (It means that the node had already 'known' about this highest parent _before_ the information
        #    from its parent was propagated, which can only happen if there is a cycle).
        #    In that case, the node's status is marked as "cycle".
        cycle_found_condition = (f.col("b." + status_col_name) != done_status_col) & (
            f.col("a." + highest_parent_id_col_name) == f.col("b." + highest_parent_id_col_name)
        )
        #    Otherwise, their "highest_parent_id" becomes the highest value between theirs and their parents.
        new_highest_parent_id = f.greatest(
            f.col("a." + highest_parent_id_col_name), f.col("b." + highest_parent_id_col_name)
        ).alias(highest_parent_id_col_name)
        # 3. If its parent is "done", it is marked as "done" too. If its parent is "cycle", it is marked as "cycle" too.
        new_status = (
            f.when(
                f.col("b." + status_col_name).isin([done_status_col, cycle_status_col]), f.col("b." + status_col_name)
            )
            .when(
                # If the parent_id does not exist as node_id, we consider that we're done
                f.col("b." + node_id_col_name).isNull(),
                done_status_col,
            )
            .when(cycle_found_condition, cycle_status_col)
            .otherwise(incomplete_status_col)
            .alias(status_col_name)
        )
        # 4. Go back to 1. until all nodes are "done".

        joined_df = joined_df.select(new_node_id_col, new_parent_id_col, new_highest_parent_id, new_status)
        joined_df = joined_df.persist()
        new_df_incomplete = joined_df.where(~status_col.isin([done_status_col, cycle_status_col]))
        do_continue = new_df_incomplete.count() > 0
        new_df_done = joined_df.where(status_col.isin([done_status_col, cycle_status_col]))

        joined_df.unpersist()
        df_incomplete = new_df_incomplete.localCheckpoint()
        df_done = df_done.union(new_df_done).localCheckpoint()

    res_df = df_done
    # res_df.show()
    # +-------+---------+-----------------+------+
    # |node_id|parent_id|highest_parent_id|status|
    # +-------+---------+-----------------+------+
    # |      3|        3|                3|     1|
    # |      2|        3|                3|     1|
    # |      1|        3|                3|     1|
    # |     12|       13|               13|    -1|
    # |     11|       13|               13|    -1|
    # |     13|       12|               13|    -1|
    # +-------+---------+-----------------+------+
    res_df = res_df.select(
        node_id_col_name,
        f.when(status_col == cycle_status_col, f.lit(None)).otherwise(parent_id_col).alias(parent_id_col_name),
        status_col,
    )
    # res_df.show()
    # +-------+---------+------+
    # |node_id|parent_id|status|
    # +-------+---------+------+
    # |      3|        3|     1|
    # |      2|        3|     1|
    # |      1|        3|     1|
    # |     12|     null|    -1|
    # |     11|     null|    -1|
    # |     13|     null|    -1|
    # +-------+---------+------+
    return res_df


def ascending_forest_traversal(
    input_df: DataFrame, node_id: str, parent_id: str, keep_labels: bool = False
) -> DataFrame:
    """Given a DataFrame representing a labeled forest with columns `id`, `parent_id` and other label columns,
    performs a graph traversal that will return a DataFrame with the same schema that gives for each node
    the labels of it's furthest ancestor.

    In the input DataFrame, a node is considered to have no parent if its parent_id is null or equal to its node_id.
    In the output DataFrame, a node that has no parent will have its parent_id equal to its node_id.
    Cycle protection: If the graph contains any cycle, the nodes in that cycle will have a NULL parent_id.

    It has a security against dependency cycles, but no security preventing
    a combinatorial explosion if some nodes have more than one parent.

    Args:
        input_df: A Spark DataFrame
        node_id: Name of the column that represent the node's ids
        parent_id: Name of the column that represent the parent node's ids
        keep_labels: If set to true, add two structs column called "node" and "furthest_ancestor" containing
            the content of the row from the input DataFrame for the corresponding nodes and their furthest ancestor

    Returns:
        A DataFrame with two columns named according to `node_id` and `parent_id` that gives for each node
        the id of it's furthest ancestor (in the `parent_id` column).
        If the option `keep_labels` is used, two extra columns of type STRUCT are a added to the output DataFrame,
        they represent the content of the rows in the input DataFrame corresponding to the node and its furthest
        ancestor, respectively.

    Examples:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()

        Given a DataFrame with pokemon attributes and evolution links

        >>> input_df = spark.sql('''
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
        >>> input_df.show()
        +----------+--------------------+------------+--------------+
        |pokemon.id|pokemon.evolve_to_id|pokemon.name| pokemon.types|
        +----------+--------------------+------------+--------------+
        |         4|                   5|  Charmander|        [Fire]|
        |         5|                   6|  Charmeleon|        [Fire]|
        |         6|                null|   Charizard|[Fire, Flying]|
        +----------+--------------------+------------+--------------+
        <BLANKLINE>

        We compute a DataFrame that for each pokemon.id gives the attributes of its highest level of evolution

        >>> ascending_forest_traversal(input_df, "pokemon.id", "pokemon.evolve_to_id").orderBy("`pokemon.id`").show()
        +----------+--------------------+
        |pokemon.id|pokemon.evolve_to_id|
        +----------+--------------------+
        |         4|                   6|
        |         5|                   6|
        |         6|                   6|
        +----------+--------------------+
        <BLANKLINE>

        With the `keep_label` option extra joins are performed at the end of the algorithm to add two struct columns
        containing the corresponding row for the original node and the furthest ancestor.

        >>> ascending_forest_traversal(input_df, "pokemon.id", "pokemon.evolve_to_id", keep_labels=True
        ...     ).orderBy("`pokemon.id`").show(10, False)
        +----------+--------------------+------------------------------------+------------------------------------+
        |pokemon.id|pokemon.evolve_to_id|node                                |furthest_ancestor                   |
        +----------+--------------------+------------------------------------+------------------------------------+
        |4         |6                   |{4, 5, Charmander, [Fire]}          |{6, null, Charizard, [Fire, Flying]}|
        |5         |6                   |{5, 6, Charmeleon, [Fire]}          |{6, null, Charizard, [Fire, Flying]}|
        |6         |6                   |{6, null, Charizard, [Fire, Flying]}|{6, null, Charizard, [Fire, Flying]}|
        +----------+--------------------+------------------------------------+------------------------------------+
        <BLANKLINE>

        *Cycle Protection:* to prevent the algorithm from looping indefinitely, cycles are detected, and the nodes
        that are part of cycles will end up with a NULL value as their furthest ancestor

        >>> input_df = spark.sql('''
        ...     SELECT
        ...       col1 as `node_id`,
        ...       col2 as `parent_id`
        ...     FROM VALUES (1, 2), (2, 3), (3, 1)
        ... ''')
        >>> input_df.show()
        +-------+---------+
        |node_id|parent_id|
        +-------+---------+
        |      1|        2|
        |      2|        3|
        |      3|        1|
        +-------+---------+
        <BLANKLINE>
        >>> ascending_forest_traversal(input_df, "node_id", "parent_id").orderBy("node_id").show()
        +-------+---------+
        |node_id|parent_id|
        +-------+---------+
        |      1|     null|
        |      2|     null|
        |      3|     null|
        +-------+---------+
        <BLANKLINE>
    """
    assert_true(
        node_id in input_df.columns, "Could not find column %s in Dataframe's columns: %s" % (node_id, input_df.columns)
    )
    assert_true(
        parent_id in input_df.columns,
        "Could not find column %s in Dataframe's columns: %s" % (parent_id, input_df.columns),
    )
    node_id_col_name = "node_id"
    parent_id_col_name = "parent_id"
    status_col_name = "status"
    df = input_df.select(
        f.col(quote(node_id)).alias(node_id_col_name), f.col(quote(parent_id)).alias(parent_id_col_name)
    )

    res_df = _ascending_forest_traversal(
        df, node_id_col_name=node_id_col_name, parent_id_col_name=parent_id_col_name, status_col_name=status_col_name
    )
    res_df = res_df.select(
        f.col(node_id_col_name).alias(node_id),
        f.col(parent_id_col_name).alias(parent_id),
    )

    if keep_labels:
        res_df = res_df.join(input_df, node_id).select(
            res_df["*"],
            f.struct(*[input_df[quote(col)] for col in input_df.columns]).alias("node"),
        )
        res_df = res_df.join(input_df, res_df[quote(parent_id)] == input_df[quote(node_id)]).select(
            res_df[quote(node_id)],
            res_df[quote(parent_id)],
            res_df["node"],
            f.struct(*[input_df[quote(col)] for col in input_df.columns]).alias("furthest_ancestor"),
        )

    return res_df
