from typing import List

from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from spark_frame.utils import assert_true, quote, quote_columns


def unpivot(df: DataFrame, pivot_columns: List[str], key_alias: str = "key", value_alias: str = "value") -> DataFrame:
    """Unpivot the given DataFrame along the specified pivot columns.
    All columns that are not pivot columns should have the same type.

    This is the inverse transformation of the [pyspark.sql.GroupedData.pivot][] operation.

    Args:
        df: A DataFrame
        pivot_columns: The list of columns names on which to perform the pivot
        key_alias: Alias given to the 'key' column
        value_alias: Alias given to the 'value' column

    Returns:
        An unpivotted DataFrame

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.createDataFrame([
        ...    (2018, "Orange",  None, 4000, None),
        ...    (2018, "Beans",   None, 1500, 2000),
        ...    (2018, "Banana",  2000,  400, None),
        ...    (2018, "Carrots", 2000, 1200, None),
        ...    (2019, "Orange",  5000, None, 5000),
        ...    (2019, "Beans",   None, 1500, 2000),
        ...    (2019, "Banana",  None, 1400,  400),
        ...    (2019, "Carrots", None,  200, None),
        ...  ], "year INT, product STRING, Canada INT, China INT, Mexico INT"
        ... )
        >>> df.show()
        +----+-------+------+-----+------+
        |year|product|Canada|China|Mexico|
        +----+-------+------+-----+------+
        |2018| Orange|  NULL| 4000|  NULL|
        |2018|  Beans|  NULL| 1500|  2000|
        |2018| Banana|  2000|  400|  NULL|
        |2018|Carrots|  2000| 1200|  NULL|
        |2019| Orange|  5000| NULL|  5000|
        |2019|  Beans|  NULL| 1500|  2000|
        |2019| Banana|  NULL| 1400|   400|
        |2019|Carrots|  NULL|  200|  NULL|
        +----+-------+------+-----+------+
        <BLANKLINE>
        >>> unpivot(df, ['year', 'product'], key_alias='country', value_alias='total').show(100)
        +----+-------+-------+-----+
        |year|product|country|total|
        +----+-------+-------+-----+
        |2018| Orange| Canada| NULL|
        |2018| Orange|  China| 4000|
        |2018| Orange| Mexico| NULL|
        |2018|  Beans| Canada| NULL|
        |2018|  Beans|  China| 1500|
        |2018|  Beans| Mexico| 2000|
        |2018| Banana| Canada| 2000|
        |2018| Banana|  China|  400|
        |2018| Banana| Mexico| NULL|
        |2018|Carrots| Canada| 2000|
        |2018|Carrots|  China| 1200|
        |2018|Carrots| Mexico| NULL|
        |2019| Orange| Canada| 5000|
        |2019| Orange|  China| NULL|
        |2019| Orange| Mexico| 5000|
        |2019|  Beans| Canada| NULL|
        |2019|  Beans|  China| 1500|
        |2019|  Beans| Mexico| 2000|
        |2019| Banana| Canada| NULL|
        |2019| Banana|  China| 1400|
        |2019| Banana| Mexico|  400|
        |2019|Carrots| Canada| NULL|
        |2019|Carrots|  China|  200|
        |2019|Carrots| Mexico| NULL|
        +----+-------+-------+-----+
        <BLANKLINE>
    """
    pivoted_columns = [(c, t) for (c, t) in df.dtypes if c not in pivot_columns]
    cols, types = zip(*pivoted_columns)

    # Check that all columns have the same type.
    assert_true(
        len(set(types)) == 1,
        ("All pivoted columns should be of the same type:\n Pivoted columns are: %s" % pivoted_columns),
    )

    # Create and explode an array of (column_name, column_value) structs
    kvs = f.explode(
        f.array(*[f.struct(f.lit(c).alias(key_alias), f.col(quote(c)).alias(value_alias)) for c in cols]),
    ).alias("kvs")

    return df.select([f.col(c) for c in quote_columns(pivot_columns)] + [kvs]).select(
        [*quote_columns(pivot_columns), "kvs.*"],
    )
