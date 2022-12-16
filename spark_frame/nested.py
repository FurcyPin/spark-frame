from collections import OrderedDict
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union, cast

from pyspark.sql import Column
from pyspark.sql import functions as f

from spark_frame import fp
from spark_frame.conf import REPETITION_MARKER, STRUCT_SEPARATOR
from spark_frame.fp import PrintableFunction, higher_order
from spark_frame.utils import assert_true, str_to_col

ColumnTransformation = Callable[[Optional[Column]], Column]
AnyKindOfTransformation = Union[str, Column, ColumnTransformation, "PrintableFunction"]
OrderedTree = Union["OrderedTree", Dict[str, Union["OrderedTree", AnyKindOfTransformation]]]  # type: ignore


def __find_first_occurrence(string, *chars: str) -> int:
    """Find the index of the first occurence of the given characters in the given string.
    Return -1 if no such occurrence is found.

    Examples:

        >>> __find_first_occurrence("a.!b", "!")
        2
        >>> __find_first_occurrence("a.!b", "!", ".")
        1
    """
    for i, c in enumerate(string):
        if c in chars:
            return i
    return -1


def __split_string_and_keep_separator(string: str, *separators: str) -> Tuple[str, Optional[str]]:
    """Split a string in half on the first occurrence of any one of the given separator.
    The separator is kept in the second half of the string.
    If the input string does not contain any of the separator, returns the string and None.

    Examples:

        >>> __split_string_and_keep_separator("a.!b", "!", ".")
        ('a', '.!b')
        >>> __split_string_and_keep_separator("a!!b", "!", ".")
        ('a', '!!b')
        >>> __split_string_and_keep_separator("ab", "!", ".")
        ('ab', None)
    """
    i = __find_first_occurrence(string, *separators)
    if i == -1:
        return string, None
    else:
        return string[:i], string[i:]


def _build_nested_struct_tree(column_transformations: Mapping[str, AnyKindOfTransformation]) -> OrderedTree:
    """Given a `Dict(column_alias, column_transformation)`, recursively build a tree structure grouping every
    common prefixes into common nodes. Structs (represented by `.`) and Arrays (represented by `!`) modifiers
    each generate a new level in the tree.

    Args:
        column_transformations: A `Dict(column_alias, column_transformation)`

    Returns:
        An ordered tree

    Examples:

        >>> _build_nested_struct_tree({
        ...   "s!.c": PrintableFunction(lambda s: s["c"], 'trans_c') ,
        ...   "s!.d": PrintableFunction(lambda s: s["d"].cast("DOUBLE"), 'trans_d'),
        ... })
        OrderedDict([('s', OrderedDict([('!', OrderedDict([('.', OrderedDict([('c', trans_c), ('d', trans_d)]))]))]))])

        >>> _build_nested_struct_tree({
        ...   "e!!.c": PrintableFunction(lambda s: s["c"], 'trans_c') ,
        ...   "e!!.d": PrintableFunction(lambda s: s["d"].cast("DOUBLE"), 'trans_d'),
        ... })  # noqa: E501
        OrderedDict([('e', OrderedDict([('!', OrderedDict([('!', OrderedDict([('.', OrderedDict([('c', trans_c), ('d', trans_d)]))]))]))]))])

        >>> _build_nested_struct_tree({
        ...   "e!": PrintableFunction(lambda e: e.cast("DOUBLE"), 'trans_e')
        ... })
        OrderedDict([('e', OrderedDict([('!', trans_e)]))])

        >>> _build_nested_struct_tree({
        ...   "e!!": PrintableFunction(lambda e: e.cast("DOUBLE"), 'trans_e')
        ... })
        OrderedDict([('e', OrderedDict([('!', OrderedDict([('!', trans_e)]))]))])
    """

    def rec_insert(node: OrderedTree, alias: str, column: AnyKindOfTransformation) -> None:
        node_col, child_col = __split_string_and_keep_separator(alias, STRUCT_SEPARATOR, REPETITION_MARKER)
        if child_col is not None and node_col == "":
            node_col = child_col[0]
            child_col = child_col[1:]
        if child_col is not None and child_col != "":
            if node_col not in node:
                node[node_col] = OrderedDict()
            rec_insert(node[node_col], child_col, column)
        else:
            node[alias] = column

    tree: OrderedTree = OrderedDict()
    for col_name, col_transformation in column_transformations.items():
        rec_insert(tree, col_name, col_transformation)
    return tree


def _convert_transformation_to_printable_function(transformation: AnyKindOfTransformation) -> PrintableFunction:
    """Transform any kind of column transformation (str, Column, Callable[[Column], Column], PrintableFunction)
    into a PrintableFunction"""
    if isinstance(transformation, PrintableFunction):
        printable_func_trans = cast(PrintableFunction, transformation)
        return PrintableFunction(
            lambda x: str_to_col(printable_func_trans(x)), lambda x: printable_func_trans.apply_alias(x)
        )
    elif callable(transformation):
        func_trans = cast(ColumnTransformation, transformation)
        return PrintableFunction(lambda x: func_trans(x), lambda x: repr(func_trans))
    elif isinstance(transformation, Column):
        col_trans = cast(Column, transformation)
        return PrintableFunction(lambda x: col_trans, lambda x: str(col_trans))
    else:
        assert_true(isinstance(transformation, str), f"Error, unsupported type: {type(transformation)}")
        str_trans = cast(str, transformation)
        return PrintableFunction(lambda x: f.col(str_trans), lambda x: f"f.col('{str_trans}')")


def _merge_functions(functions: List[PrintableFunction]) -> PrintableFunction:
    """Merge a list of column expressions or functions that each "take a struct `s` and return a column"
    into a single function that "takes a struct `s` and returns a list containing the result of each function or
    the fixed column expressions"

    In other term, given a list of functions `[f1, f2, c1, c2, ...]`,
    this returns a function `s -> [f1(s), f2(s), c1, c2, ...]`
    """
    return PrintableFunction(
        lambda x: [fun(x) for fun in functions],
        lambda x: "[" + ", ".join([fun.apply_alias(x) for fun in functions]) + "]",
    )


def _build_transformation_from_tree(root: OrderedTree, sort: bool = False) -> PrintableFunction:
    """From an intermediary abstract tree, build a PrintableFunction that produces a list of Column expressions.
    Arrays will be sorted if `sort` is set to true.

    The transformation generated by this function can be displayed as a string for debugging purposes.

    !!! Warning
        Arrays containing sub-elements of type Map cannot be sorted. When using this option, one must make sure
        that all Maps have been cast to Array<Struct> with [functions.map_entries](pyspark.sql.functions.map_entries)

    Args:
        root: The root of the abstract tree
        sort: If set to true, all arrays will be automatically sorted.

    Returns:
        A PrintableFunction that produces a list of Column expressions
    """

    def recurse_node_with_multiple_items(node: OrderedTree) -> List[PrintableFunction]:
        return [recurse_item(node, key, col_or_children) for key, col_or_children in node.items()]

    def recurse_node_with_one_item(node: OrderedTree) -> PrintableFunction:
        assert_true(len(node) == 1, "Error, this should not happen: non-struct node with more than one child")
        key, col_or_children = next(iter(node.items()))
        return recurse_item(node, key, col_or_children)

    def recurse_item(
        node: OrderedTree, key: str, col_or_children: Union[AnyKindOfTransformation, OrderedTree]
    ) -> PrintableFunction:
        is_struct = key == STRUCT_SEPARATOR
        is_repeated = key == REPETITION_MARKER
        has_children = isinstance(col_or_children, Dict)
        if is_struct:
            assert_true(len(node) == 1, "Error, this should not happen: tree node of type struct with siblings")
            assert_true(has_children, "Error, this should not happen: struct without children")
            children_transformations = recurse_node_with_multiple_items(col_or_children)
            merged_transformation = _merge_functions(children_transformations)
            res = fp.compose(higher_order.struct, merged_transformation)
            return res
        if is_repeated:
            assert_true(len(node) == 1, "Error, this should not happen: tree node of type array with siblings")
            transform_col = recurse_node_with_one_item(col_or_children) if has_children else col_or_children
            res = higher_order.transform(cast(Callable, transform_col))
            if sort:
                res = fp.compose(higher_order.sort_array, res)
            return res
        if has_children:
            child_transformation = recurse_node_with_one_item(col_or_children)
            col = fp.compose(child_transformation, higher_order.safe_struct_get(key))
        else:
            col = _convert_transformation_to_printable_function(cast(AnyKindOfTransformation, col_or_children))
        col = fp.compose(higher_order.alias(key), col)
        return col

    root_transformations = list(recurse_node_with_multiple_items(root))
    merged_root_transformation = _merge_functions(root_transformations)
    return merged_root_transformation


def resolve_nested_columns(columns: Mapping[str, AnyKindOfTransformation], sort: bool = False) -> List[Column]:
    """Builds a list of column expressions to manipulate structs and repeated records

    >>> from pyspark.sql import SparkSession
    >>> from pyspark.sql import functions as f
    >>> spark = SparkSession.builder.appName("doctest").getOrCreate()

    >>> df = spark.sql('''
    ...     SELECT INLINE(ARRAY(
    ...       STRUCT(ARRAY(ARRAY(1), ARRAY(2, 3)) as e)
    ...     ))
    ... ''')
    >>> df.printSchema()
    root
     |-- e: array (nullable = false)
     |    |-- element: array (containsNull = false)
     |    |    |-- element: integer (containsNull = false)
    <BLANKLINE>
    >>> df.show(truncate=False)
    +-------------+
    |e            |
    +-------------+
    |[[1], [2, 3]]|
    +-------------+
    <BLANKLINE>
    >>> res_df = df.select(*resolve_nested_columns({
    ...     "e!!": lambda e: e.cast("DOUBLE"),
    ... }, sort=True))
    >>> res_df.printSchema()
    root
     |-- e: array (nullable = false)
     |    |-- element: array (containsNull = false)
     |    |    |-- element: double (containsNull = false)
    <BLANKLINE>
    >>> res_df.show(truncate=False)
    +-------------------+
    |e                  |
    +-------------------+
    |[[1.0], [2.0, 3.0]]|
    +-------------------+
    <BLANKLINE>

    Args:
        columns: A mapping (column_name -> function to apply to this column)
        sort: If set to true, all arrays will be automatically sorted.

    Returns:

    """
    tree = _build_nested_struct_tree(columns)
    root_transformation = _build_transformation_from_tree(tree, sort)
    return root_transformation(None)
