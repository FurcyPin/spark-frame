from collections import OrderedDict
from typing import Dict, List, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from spark_frame.data_type_utils import is_struct
from spark_frame.transformations_impl.flatten import flatten
from spark_frame.utils import quote

OrderedTree = Union["OrderedTree", Dict[str, "OrderedTree"]]  # type: ignore


def _build_nested_struct_tree(columns: List[str], struct_separator: str) -> OrderedTree:
    """Given a list of flattened column names and a separator

    >>> _build_nested_struct_tree(["id", "s.a", "s.b.c", "s.b.d"], ".")
    OrderedDict([('id', None), ('s', OrderedDict([('a', None), ('b', OrderedDict([('c', None), ('d', None)]))]))])

    :param columns: Name of the flattened columns
    :param struct_separator: Separator used in the column names for structs
    :return:
    """

    def rec_insert(node: OrderedTree, col: str) -> None:
        if struct_separator in col:
            struct, subcol = col.split(struct_separator, 1)
            if struct not in node:
                node[struct] = OrderedDict()
            rec_insert(node[struct], subcol)
        else:
            node[col] = None

    tree: OrderedTree = OrderedDict()
    for c in columns:
        rec_insert(tree, c)
    return tree


def _build_struct_from_tree(node: OrderedTree, separator: str, prefix: str = "") -> List[Column]:
    """Given an intermediate tree representing a nested struct, build a Spark Column
    that represents this nested structure.

    >>> from pyspark.sql import SparkSession
    >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
    >>> tree = OrderedDict([('b!', OrderedDict([
    ...      ('c', None),
    ...      ('d', None)
    ...    ]))])
    >>> _build_struct_from_tree(tree, ".") # noqa: E501
    [Column<'CASE WHEN ((true AND (`b!.c` AS c IS NULL)) AND (`b!.d` AS d IS NULL)) THEN NULL ELSE struct(`b!.c` AS c, `b!.d` AS d) END AS `b!`'>]

    :param node:
    :param separator:
    :param prefix:
    :return:
    """
    cols = []
    for key, value in node.items():
        if value is None:
            cols.append(f.col(quote(prefix + key)).alias(key))
        else:
            fields = _build_struct_from_tree(value, separator, prefix + key + separator)
            # We don't want structs where all fields are null so we check for this
            all_fields_are_null = f.lit(True)
            for field in fields:
                all_fields_are_null = all_fields_are_null & f.isnull(field)

            struct_col = f.when(all_fields_are_null, f.lit(None)).otherwise(f.struct(*fields)).alias(key)
            cols.append(struct_col)
    return cols


def unflatten(df: DataFrame, separator: str = ".") -> DataFrame:
    """Reverse of the flatten operation
    Nested fields names will be separated from each other using the specified separator

    Args:
        df: A Spark DataFrame
        separator: A string used to separate the structs names from their elements.
                   It might be useful to change the separator when some DataFrame's column names already contain dots

    Returns:
        A flattened DataFrame

    Examples:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.createDataFrame([(1, 1, 1, 1)], "id INT, `s.a` INT, `s.b.c` INT, `s.b.d` INT")
        >>> df.printSchema()
        root
         |-- id: integer (nullable = true)
         |-- s.a: integer (nullable = true)
         |-- s.b.c: integer (nullable = true)
         |-- s.b.d: integer (nullable = true)
        <BLANKLINE>
        >>> unflatten(df).printSchema()
        root
         |-- id: integer (nullable = true)
         |-- s: struct (nullable = true)
         |    |-- a: integer (nullable = true)
         |    |-- b: struct (nullable = true)
         |    |    |-- c: integer (nullable = true)
         |    |    |-- d: integer (nullable = true)
        <BLANKLINE>
        >>> df = spark.createDataFrame([(1, 1, 1)], "id INT, `s.s1?a.a1` INT, `s.s1?b.b1` INT")
        >>> df.printSchema()
        root
         |-- id: integer (nullable = true)
         |-- s.s1?a.a1: integer (nullable = true)
         |-- s.s1?b.b1: integer (nullable = true)
        <BLANKLINE>
        >>> unflatten(df, "?").printSchema()
        root
         |-- id: integer (nullable = true)
         |-- s.s1: struct (nullable = true)
         |    |-- a.a1: integer (nullable = true)
         |    |-- b.b1: integer (nullable = true)
        <BLANKLINE>
    """
    # The idea is to recursively write a "SELECT struct(a, struct(s.b.c, s.b.d)) as s" for each nested column.
    # There is a little twist as we don't want to rebuild the struct if all its fields are null, so we add a CASE WHEN

    def has_structs(df: DataFrame) -> bool:
        struct_fields = [field for field in df.schema if is_struct(field)]
        return len(struct_fields) > 0

    if has_structs(df):
        df = flatten(df)

    tree = _build_nested_struct_tree(df.columns, separator)
    cols = _build_struct_from_tree(tree, separator)
    return df.select(cols)
