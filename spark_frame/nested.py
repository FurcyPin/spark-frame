from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, cast

from pyspark.sql import Column
from pyspark.sql import functions as f

from spark_frame.conf import REPETITION_MARKER, STRUCT_SEPARATOR
from spark_frame.utils import StringOrColumn, assert_true, safe_struct_get, str_to_col

ColumnTransformation = Callable[[Optional[Column]], Column]
AnyKindOfTransformation = Union[str, Column, ColumnTransformation, "PrintableFunction"]
OrderedTree = Union["OrderedTree", Dict[str, Union["OrderedTree", AnyKindOfTransformation]]]  # type: ignore


class PrintableFunction:
    """Wrapper for anonymous functions with a short description making them much human-friendly when printed.

    Very useful when debugging, useless otherwise.

    Args:
        func: A function that takes a Column and return a Column.
        alias: A string or a function that takes a string and return a string.

    Examples:

        >>> print(PrintableFunction(lambda s: s["c"], "my_function"))
        my_function

        >>> print(PrintableFunction(lambda s: s["c"], lambda s: f'{s}["c"]'))
        lambda x: x["c"]

        >>> func = PrintableFunction(lambda s: s.cast("Double"), lambda s: f'{s}.cast("Double")')
        >>> print(func.alias("s"))
        s.cast("Double")

        Composition:

        >>> f1 = PrintableFunction(lambda s: s.cast("Double"), lambda s: f'{s}.cast("Double")')
        >>> f2 = PrintableFunction(lambda s: s * s, lambda s: f'{f"({s} * {s})"}')
        >>> f2_then_f1 = PrintableFunction(lambda s: f1(f2(s)), lambda s: f1.alias(f2.alias(s)))
        >>> print(f2_then_f1)
        lambda x: (x * x).cast("Double")

    """

    def __init__(self, func: Callable[[Any], Any], alias: Union[str, Callable[[str], str]]) -> None:
        self.func: Callable[[Any], Any] = func
        self.alias: Union[str, Callable[[str], str]] = alias

    def __repr__(self) -> str:
        if callable(self.alias):
            return f"lambda x: {cast(Callable, self.alias)('x')}"
        else:
            return cast(str, self.alias)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __instancecheck__(self, instance):
        return isinstance(self.func, instance)


class HigherOrder:

    identity = PrintableFunction(lambda s: s, lambda s: s)

    @staticmethod
    def _safe_struct_get_alias(s: str, field: str) -> str:
        """Return a string representation for the `safe_struct_get` method."""
        if s is None:
            return f"f.col('{field}')"
        else:
            return f"{s}[{field}]"

    @staticmethod
    def safe_struct_get(key: str):
        return PrintableFunction(
            lambda s: safe_struct_get(s, key), lambda s: HigherOrder._safe_struct_get_alias(s, key)
        )

    @staticmethod
    def compose(f1: PrintableFunction, f2: PrintableFunction) -> PrintableFunction:
        if callable(f1.alias) and callable(f2.alias):
            c1 = cast(Callable, f1.alias)
            c2 = cast(Callable, f2.alias)
            return PrintableFunction(lambda s: f1.func(f2.func(s)), lambda s: c1(c2(s)))
        elif callable(f1.alias) and not callable(f2.alias):
            c1 = cast(Callable, f1.alias)
            a2 = str(f2.alias)
            return PrintableFunction(lambda s: f1.func(f2.func(s)), c1(a2))
        elif not callable(f1.alias) and callable(f2.alias):
            a1 = str(f1.alias)
            c2 = cast(Callable, f2.alias)
            return PrintableFunction(lambda s: f1.func(f2.func(s)), lambda s: f"{a1}({c2(s)})")
        else:
            a1 = str(f1.alias)
            a2 = str(f2.alias)
            return PrintableFunction(lambda s: f1.func(f2.func(s)), f"{a1}({a2})")


def __get_func(function: Union[ColumnTransformation, PrintableFunction]) -> ColumnTransformation:
    """Given a ColumnTransformation, printable or not, return a ColumnTransformation"""
    if isinstance(function, PrintableFunction):
        return function.func
    else:
        return function


def __get_alias(function: Union[ColumnTransformation, PrintableFunction], x: str) -> str:
    """Given a ColumnTransformation, printable or not, return a string representation of it"""
    if isinstance(function, PrintableFunction) and callable(function.alias):
        return function.alias(x)
    else:
        return repr(function)


def __str_to_col_alias(args: StringOrColumn) -> str:
    """Return a string representation for the `str_to_col` method."""
    if isinstance(args, str):
        return f"f.col('{args}')"
    else:
        return str(args)


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


def _merge_functions(functions: List[Tuple[AnyKindOfTransformation, str]]) -> PrintableFunction:
    """Merge a list of column expressions or functions that each "take a struct `s` and return a column"
    into a single function that "takes a struct `s` and returns a list containing the result of each function or
    the fixed column expressions"

    In other term, given a list of functions `[f1, f2, c1, c2, ...]`,
    this returns a function `s -> [f1(s), f2(s), c1, c2, ...]`
    """

    def aux(s: Optional[Column], fun: AnyKindOfTransformation, alias: str) -> Column:
        if alias is None:
            return fun(s)
        if callable(fun):
            return str_to_col(fun(s)).alias(alias)
        else:
            fun = cast(Union[str, Column], fun)
            return str_to_col(fun).alias(alias)

    def aux_alias(s: str, fun: AnyKindOfTransformation, alias: str) -> str:
        if alias is None:
            return __get_alias(fun, s)
        if callable(fun):
            return __get_alias(fun, s) + f".alias('{alias}')"
        else:
            fun = cast(Union[str, Column], fun)
            return f"{__str_to_col_alias(fun)}.alias('{alias}')"

    return PrintableFunction(
        lambda x: [aux(x, fun, alias) for fun, alias in functions],
        lambda x: "[" + ", ".join([f"{aux_alias(x, fun, alias)}" for fun, alias in functions]) + "]",
    )


def _build_struct_from_tree(node: OrderedTree, sort: bool = False) -> List[Column]:
    """Build a list of Column expression from an intermediary abstract tree.
    Arrays will be sorted if `sort` is set to true.

    !!! Warning
        Arrays containing sub-elements of type Map cannot be sorted. When using this option, one must make sure
        that all Maps have been cast to Array<Struct> with [functions.map_entries](pyspark.sql.functions.map_entries)

    Args:
        node: The root of the abstract tree
        sort: If set to true, all arrays will be automatically sorted.

    Returns:
        A list of Column expressions

    """

    def recurse(node: OrderedTree) -> PrintableFunction:
        cols: List[Tuple[AnyKindOfTransformation, str]] = []
        for key, col_or_children in node.items():
            is_repeated = key == REPETITION_MARKER
            is_struct = key == STRUCT_SEPARATOR
            has_children = isinstance(col_or_children, Dict)

            if is_struct:
                assert_true(len(node) == 1, "Error, this should not happen: tree node of type struct with siblings")
                assert_true(has_children, "Error, this should not happen: struct without children")
                child_transformation = recurse(col_or_children)
                res = PrintableFunction(
                    lambda x: f.struct(*child_transformation(x)),
                    lambda x: f"f.struct(*{__get_alias(child_transformation, x)})",
                )
                return res

            if is_repeated:
                # TODO: add upstream check for this case
                transform_col = recurse(col_or_children) if has_children else col_or_children
                assert_true(
                    callable(transform_col),
                    "Error, this should not happen: non-callable transformation on repeated node",
                )
                assert_true(len(node) == 1, "Error, this should not happen: tree node of type struct with siblings")
                func_col = cast(Callable, transform_col)
                unsorted_col = PrintableFunction(
                    lambda x: f.transform(x, __get_func(func_col)), lambda x: f"f.transform({x}, {func_col})"
                )
                if sort:
                    res = PrintableFunction(
                        lambda x: f.sort_array(unsorted_col(x)),
                        lambda x: f"f.sort_array(unsorted_col({x}))",
                    )
                else:
                    res = unsorted_col
                return res

            col: AnyKindOfTransformation
            if has_children:
                child_transformation = recurse(col_or_children)
                col = HigherOrder.compose(child_transformation, HigherOrder.safe_struct_get(key))
            else:
                col = col_or_children
            cols.append((col, key))
        return _merge_functions(cols)

    root_transformation: PrintableFunction = cast(PrintableFunction, recurse(node))
    # The transformation by this function can be debugged by printing `root_transformation.alias(None)`
    return root_transformation(None)


def resolve_nested_columns(columns: Mapping[str, ColumnTransformation], sort: bool = False) -> List[Column]:
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
    return _build_struct_from_tree(tree, sort)
