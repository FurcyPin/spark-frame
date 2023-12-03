from collections import OrderedDict
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import ArrayType, DataType, MapType, StructType

from spark_frame import fp
from spark_frame.conf import (
    MAP_KEY,
    MAP_MARKER,
    MAP_VALUE,
    REPETITION_MARKER,
    STRUCT_SEPARATOR,
)
from spark_frame.fp import PrintableFunction, higher_order
from spark_frame.utils import (
    AnalysisException,
    assert_true,
    group_by_key,
    is_direct_sub_field_of_any,
    quote,
    substring_before_last_occurrence,
)

ColumnTransformation = Callable[[Optional[Column]], Column]
AnyKindOfTransformation = Union[
    str, Column, ColumnTransformation, "PrintableFunction", None
]
OrderedTree = Union["OrderedTree", Dict[str, Union["OrderedTree", Optional[AnyKindOfTransformation]]]]  # type: ignore


def _identity_column_transformation(col: Column, data_type: DataType) -> Column:
    return col


def build_transformation_from_schema(
    schema: StructType,
    column_transformation: Optional[
        Callable[[Column, DataType], Optional[Column]]
    ] = None,
    name_transformation: Optional[Callable[[str], str]] = None,
) -> PrintableFunction:
    """Given a DataFrame schema, recursively build a PrintableFunction that reproduces this schema.
    Optional transformations may be applied.

    This method works recursively on structs, arrays and maps.

    The transformation generated by this function can be displayed as a string for debugging purposes.

    Args:
        schema: A DataFrame schema
        column_transformation: Transformation to apply to all fields of the output result
        name_transformation: Transformation to apply to all field names of the output result

    Returns:
        A PrintableFunction that produces a list of Column expressions
    """
    if column_transformation is None:
        _column_transformation: Callable[
            [Column, DataType], Optional[Column]
        ] = _identity_column_transformation
    else:
        _column_transformation = column_transformation
    if name_transformation is None:
        _name_transformation: Callable[[str], str] = higher_order.identity
    else:
        _name_transformation = name_transformation

    def column_transformation_with_fallback(col: Column, data_type: DataType) -> Column:
        """Enrich the input transformation with a fallback that
        returns the input if the result of the transformation is None
        """
        result = _column_transformation(col, data_type)
        if result is None:
            return col
        else:
            return result

    def recurse_data_type(
        data_type: DataType, parent_structs: List[str]
    ) -> PrintableFunction:
        if isinstance(data_type, StructType):
            children_transformations = list(
                recurse_struct_type(data_type, parent_structs)
            )
            res = _merge_functions(children_transformations)
            res = fp.compose(higher_order.struct, res)
        elif isinstance(data_type, ArrayType):
            element_transformation = recurse_data_type(
                data_type.elementType, parent_structs=[]
            )
            res = higher_order.transform(element_transformation)
            res = fp.compose(res, higher_order.recursive_struct_get(parent_structs))
        elif isinstance(data_type, MapType):
            key_transformation = recurse_data_type(data_type.keyType, parent_structs=[])
            value_transformation = recurse_data_type(
                data_type.valueType, parent_structs=[]
            )
            f1 = higher_order.transform_keys(key_transformation)
            f2 = higher_order.transform_values(value_transformation)
            f3 = higher_order.recursive_struct_get(parent_structs)
            res = fp.compose(f1, f2, f3)
        else:
            res = higher_order.recursive_struct_get(parent_structs)
        col_transformation = PrintableFunction(
            lambda c: column_transformation_with_fallback(c, data_type),
            lambda s: str(s),
        )
        return fp.compose(col_transformation, res)

    def recurse_struct_type(
        struct: StructType, parent_structs: List[str]
    ) -> Generator[PrintableFunction, None, None]:
        for field in struct:
            field_transformation = recurse_data_type(
                field.dataType, parent_structs + [field.name]
            )
            res = fp.compose(
                higher_order.alias(_name_transformation(field.name)),
                field_transformation,
            )
            yield res

    root_transformations = list(recurse_struct_type(schema, parent_structs=[]))
    merged_root_transformation = _merge_functions(root_transformations)
    return merged_root_transformation


def __find_first_occurrence(string: str, *chars: str) -> int:
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


def _split_string_and_keep_separator(
    string: str, *separators: str
) -> Tuple[str, Optional[str]]:
    """Split a string in half on the first occurrence of any one of the given separator.
    The separator is kept in the second half of the string.
    If the input string does not contain any of the separator, returns the string and None.

    Examples:
        >>> _split_string_and_keep_separator("a.!b", "!", ".")
        ('a', '.!b')
        >>> _split_string_and_keep_separator("a!!b", "!", ".")
        ('a', '!!b')
        >>> _split_string_and_keep_separator("ab", "!", ".")
        ('ab', None)
    """
    i = __find_first_occurrence(string, *separators)
    if i == -1:
        return string, None
    else:
        return string[:i], string[i:]


def _deepest_granularity(field_name: str) -> List[str]:
    """Return the part of a field_name corresponding to it's deepest granularity

    Examples:
        >>> _deepest_granularity("a")
        ['a']
        >>> _deepest_granularity("a!")
        []
        >>> _deepest_granularity("a!.b.c")
        ['b', 'c']
    """
    return [
        s
        for s in field_name.split(REPETITION_MARKER)[-1].split(STRUCT_SEPARATOR)
        if s != ""
    ]


def _build_nested_struct_tree(
    column_transformations: Mapping[str, Optional[AnyKindOfTransformation]]
) -> OrderedTree:
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
        ... })
        OrderedDict([('e', OrderedDict([('!', OrderedDict([('!', OrderedDict([('.', OrderedDict([('c', trans_c), ('d', trans_d)]))]))]))]))])

        >>> _build_nested_struct_tree({
        ...   "e!": PrintableFunction(lambda e: e.cast("DOUBLE"), 'trans_e')
        ... })
        OrderedDict([('e', OrderedDict([('!', trans_e)]))])

        >>> _build_nested_struct_tree({
        ...   "e!!": PrintableFunction(lambda e: e.cast("DOUBLE"), 'trans_e')
        ... })
        OrderedDict([('e', OrderedDict([('!', OrderedDict([('!', trans_e)]))]))])

        >>> _build_nested_struct_tree({
        ...   "m1%key": PrintableFunction(lambda key : f.upper(key), 'trans_key'),
        ...   "m1%value.a": PrintableFunction(lambda value : value["a"].cast("DOUBLE"), 'trans_value')
        ... })
        OrderedDict([('m1', OrderedDict([('%', OrderedDict([('%key', trans_key), ('%value', OrderedDict([('.', OrderedDict([('a', trans_value)]))]))]))]))])
    """  # noqa: E501

    def rec_insert(
        node: OrderedTree,
        alias: str,
        column: Optional[AnyKindOfTransformation],
        is_map: bool = False,
    ) -> None:
        node_col, child_col = _split_string_and_keep_separator(
            alias, STRUCT_SEPARATOR, REPETITION_MARKER, MAP_MARKER
        )
        if child_col is not None and node_col == "":
            node_col = child_col[0]
            child_col = child_col[1:]
        if is_map:
            node_col = MAP_MARKER + node_col
            alias = MAP_MARKER + alias
        if child_col is not None and child_col != "":
            if node_col not in node:
                node[node_col] = OrderedDict()
            rec_insert(node[node_col], child_col, column, is_map=node_col == MAP_MARKER)
        else:
            node[alias] = column

    tree: OrderedTree = OrderedDict()
    for col_name, col_transformation in column_transformations.items():
        rec_insert(tree, col_name, col_transformation)
    return tree


def _convert_transformation_to_printable_function(
    transformation: AnyKindOfTransformation,
    parent_structs: List[str],
) -> PrintableFunction:
    """Transform any kind of column transformation (str, Column, Callable[[Column], Column], PrintableFunction)
    into a PrintableFunction.

    The transformation is "boxed" which means that they will be passed an array containing all eligible arguments.
    Only the "n" right-most arguments will be passed to the transformation to match it's arity.

    If a transformations returns a string, we assume it is meant as a Column name and try to convert it into a Column.
    """
    if isinstance(transformation, PrintableFunction):
        printable_func_trans = transformation
        res = printable_func_trans
    elif callable(transformation):
        func_trans = transformation
        res = PrintableFunction(func_trans, lambda x: repr(func_trans))
    elif isinstance(transformation, Column):
        col_trans = transformation
        res = PrintableFunction(lambda x: col_trans, lambda x: str(col_trans))
    elif isinstance(transformation, str):
        str_trans = transformation
        res = PrintableFunction(
            lambda x: f.col(str_trans), lambda x: f"f.col('{str_trans}')"
        )
    else:
        assert_true(
            transformation is None,
            f"Error, unsupported transformation type: {type(transformation)}",
        )
        res = higher_order.recursive_struct_get(parent_structs)

    # The transformation is "boxed" which means that they will be passed an array containing all eligible arguments.
    # Only the "n" right-most arguments will be passed to the transformation to match it's arity.
    res = res.boxed()
    # If a transformations returns a string, we assume it is meant as a Column name and try to convert it into a Column.
    res = fp.compose(higher_order.str_to_col, res)
    return res


def _merge_functions(functions: List[PrintableFunction]) -> PrintableFunction:
    """Merge a list of column expressions or functions that each "take a struct `s` and return a column"
    into a single function that "takes a struct `s` and returns a list containing the result of each function or
    the fixed column expressions"

    In other term, given a list of functions `[f1, f2, c1, c2, ...]`,
    this returns a function `s -> [f1(s), f2(s), c1, c2, ...]`
    """
    return PrintableFunction(
        lambda x: [fun(x) for fun in functions],
        lambda x: "[" + ", ".join([fun.alias(x) for fun in functions]) + "]",
    )


def _build_transformation_from_tree(root: OrderedTree) -> PrintableFunction:
    """From an intermediary abstract tree, build a PrintableFunction that produces a list of Column expressions.

    The transformation generated by this function can be displayed as a string for debugging purposes.

    Args:
        root: The root of the abstract tree

    Returns:
        A PrintableFunction that produces a list of Column expressions
    """

    def recurse_node_with_multiple_items(
        node: OrderedTree, parent_structs: List[str]
    ) -> List[PrintableFunction]:
        return [
            recurse_item(node, key, col_or_children, parent_structs)
            for key, col_or_children in node.items()
        ]

    def recurse_node_with_one_item(
        col_or_children: Union[AnyKindOfTransformation, OrderedTree],
        parent_structs: List[str],
    ) -> PrintableFunction:
        has_children = isinstance(col_or_children, Dict)
        if has_children:
            node = cast(Dict, col_or_children)
            assert_true(
                len(node) == 1,
                "Error, this should not happen: non-struct node with more than one child",
            )
            key, col_or_children = next(iter(node.items()))
            return recurse_item(node, key, col_or_children, parent_structs)
        else:
            return _convert_transformation_to_printable_function(
                cast(AnyKindOfTransformation, col_or_children),
                parent_structs,
            )

    def recurse_item(
        node: OrderedTree,
        key: str,
        col_or_children: Union[AnyKindOfTransformation, OrderedTree],
        parent_structs: List[str],
    ) -> PrintableFunction:
        if key == STRUCT_SEPARATOR:
            assert_true(
                len(node) == 1,
                "Error, this should not happen: tree node of type struct with siblings",
            )
            has_children = isinstance(col_or_children, Dict)
            assert_true(
                has_children, "Error, this should not happen: struct without children"
            )
            child_transformations = recurse_node_with_multiple_items(
                col_or_children, parent_structs
            )
            merged_transformation = _merge_functions(child_transformations)
            res = fp.compose(higher_order.struct, merged_transformation)
            return res
        elif key == REPETITION_MARKER:
            assert_true(
                len(node) == 1,
                "Error, this should not happen: tree node of type array with siblings",
            )
            repeated_col = recurse_node_with_one_item(
                col_or_children, parent_structs=[]
            )
            res = higher_order.boxed_transform(repeated_col, parent_structs)
            return res
        elif key == MAP_MARKER:
            [
                key_transformation,
                value_transformation,
            ] = recurse_node_with_multiple_items(
                col_or_children,
                parent_structs=[],
            )
            res = higher_order.boxed_transform_map(
                key_transformation, value_transformation, parent_structs
            )
            return res
        elif key in [MAP_MARKER + MAP_KEY, MAP_MARKER + MAP_VALUE]:
            child_transformation = recurse_node_with_one_item(
                col_or_children, parent_structs=[]
            )
            return child_transformation
        else:
            child_transformation = recurse_node_with_one_item(
                col_or_children, parent_structs + [key]
            )
            col = fp.compose(higher_order.alias(key), child_transformation)
            return col

    root_transformations = list(
        recurse_node_with_multiple_items(root, parent_structs=[])
    )
    merged_root_transformation = _merge_functions(root_transformations)
    return merged_root_transformation


def validate_field_marker_followed_by_non_struct_character(
    field_name: str,
) -> Generator[str, None, None]:
    """Check for the following error:

    - Repeated field marker `!` followed by something else than a struct `.` separator

    Examples:
        >>> list(validate_field_marker_followed_by_non_struct_character("a!.b"))
        []
        >>> list(validate_field_marker_followed_by_non_struct_character("a!"))
        []
        >>> list(validate_field_marker_followed_by_non_struct_character("a!b"))
        ["Invalid field name 'a!b': '!' not followed by a '.'"]
    """
    for field_part in field_name.split(REPETITION_MARKER)[1:]:
        if len(field_part) > 0 and field_part[0] != STRUCT_SEPARATOR:
            yield f"Invalid field name '{field_name}': '{REPETITION_MARKER}' not followed by a '{STRUCT_SEPARATOR}'"
            return


def validate_map_marker_followed_by_non_key_value(
    field_name: str,
) -> Generator[str, None, None]:
    """Check for the following error:

    - Repeated field marker `!` followed by something else than a struct `.` separator

    Examples:
        >>> list(validate_map_marker_followed_by_non_key_value("a!.m%key"))
        []
        >>> list(validate_map_marker_followed_by_non_key_value("a!.m%value"))
        []
        >>> list(validate_map_marker_followed_by_non_key_value("a!.m%"))
        ["Invalid field name 'a!.m%': '%' not followed by a 'key' or 'value'"]
        >>> list(validate_map_marker_followed_by_non_key_value("a!.m%bad_key"))
        ["Invalid field name 'a!.m%bad_key': '%' not followed by a 'key' or 'value'"]
        >>> list(validate_map_marker_followed_by_non_key_value("a!.m%key_bad"))
        ["Invalid field name 'a!.m%key_bad': '%' not followed by a 'key' or 'value'"]
    """

    def build_message() -> str:
        return f"Invalid field name '{field_name}': '{MAP_MARKER}' not followed by a '{MAP_KEY}' or '{MAP_VALUE}'"

    for field_part in field_name.split(MAP_MARKER)[1:]:
        if len(field_part) == 0 or field_part.split(STRUCT_SEPARATOR)[0] not in [
            MAP_KEY,
            MAP_VALUE,
        ]:
            yield build_message()
            return


def validate_no_map(field_name: str) -> Generator[str, None, None]:
    """Check for the following error:

    - Map field marker `%` used when maps are not allowed

    Examples:
        >>> list(validate_no_map("a!.b"))
        []
        >>> list(validate_no_map("a!.m%key"))
        ["Invalid field name 'a!.m%key': maps are not supported for this type of transformation."]
        >>> list(validate_no_map("a!.m%value"))
        ["Invalid field name 'a!.m%value': maps are not supported for this type of transformation."]
    """

    def build_message() -> str:
        return f"Invalid field name '{field_name}': maps are not supported for this type of transformation."

    if MAP_MARKER in field_name:
        yield build_message()


def _get_prefixes_of_repeated_field(
    repeated_field: str, separator: str
) -> Generator[str, None, None]:
    """
    >>> list(_get_prefixes_of_repeated_field("a!.b!.c", separator="!"))
    ['a!', 'a!.b!']
    >>> list(_get_prefixes_of_repeated_field("a.b.c", separator="!"))
    []
    """
    prefix = ""
    for part in repeated_field.split(separator)[:-1]:
        prefix += part + separator
        yield prefix


def _get_repeated_fields(fields: List[str]) -> Set[str]:
    """
    >>> sorted(list(_get_repeated_fields(["s1!.a!", "s1!.b", "s2!.c", "s3!!!", "s4.a.b"])))
    ['s1!', 's1!.a!', 's2!', 's3!', 's3!!', 's3!!!']
    """
    return {
        prefix
        for field in fields
        for prefix in _get_prefixes_of_repeated_field(
            field, separator=REPETITION_MARKER
        )
    }


def _get_map_fields(fields: List[str]) -> Set[str]:
    """
    >>> sorted(list(_get_map_fields(["m1%key.a", "m1%key.b%key", "m1%key.b%value", "m1%value"])))
    ['m1%', 'm1%key.b%']
    """
    return {
        prefix
        for field in fields
        for prefix in _get_prefixes_of_repeated_field(field, separator=MAP_MARKER)
    }


def _find_fields_starting_with_prefix(
    prefix: str, fields: Iterable[str], separator: str
) -> List[str]:
    """Given a prefix, find in the list all field names that start with this prefix and contain exactly one more
    exclamation mark.

    Args:
        prefix: The prefix to search for in the field names
        fields: The list of field names to search through

    Returns:
        A list of field names that start with the given prefix and contain exactly one more exclamation mark

    Examples:
        >>> _find_fields_starting_with_prefix(
        ...     "a!.b!",
        ...     ["a!.b!.c!", "x!.b!.c!", "a!.b!.d!", "a!.b!.d!.e!"],
        ...     separator="!"
        ... )
        ['a!.b!.c!', 'a!.b!.d!']

        >>> _find_fields_starting_with_prefix("a%key", {"a%", "a%key.b%", "a%value.c%"}, separator="%")
        ['a%key.b%']
    """

    def aux() -> Generator[str, None, None]:
        for field in fields:
            if (
                field.startswith(prefix)
                and field.count(separator) == prefix.count(separator) + 1
            ):
                yield field

    return list(aux())


def validate_is_repeated_field_known(
    field_name: str, known_repeated_fields: Set[str]
) -> Generator[str, None, None]:
    """Check for the following error:

    - Repeated field name not matching any known field

    Examples:
        >>> list(validate_is_repeated_field_known("a!.c", {"a!", "a!.b!"}))
        []
        >>> list(validate_is_repeated_field_known("a!.c!", {"a!", "a!.b!"}))
        ["Repeated field 'a!.c!' does not exist: Did you mean one of the following? [a!.b!];"]
        >>> list(validate_is_repeated_field_known("a!.c", {"b!", "b!.c!"}))
        ["Repeated field 'a!' does not exist: Did you mean one of the following? [b!];"]
    """

    def build_message(prefix: str, last_valid_prefix: str) -> str:
        candidates = _find_fields_starting_with_prefix(
            last_valid_prefix, known_repeated_fields, REPETITION_MARKER
        )
        return (
            f"Repeated field '{prefix}' does not exist: "
            f"Did you mean one of the following? [{', '.join(candidates)}];"
        )

    prefix = ""
    for part in field_name.split(REPETITION_MARKER)[:-1]:
        last_valid_prefix = prefix
        prefix += part + REPETITION_MARKER
        if prefix not in known_repeated_fields:
            yield build_message(prefix, last_valid_prefix)
            return


def validate_is_map_field_known(
    field_name: str, known_map_fields: Set[str]
) -> Generator[str, None, None]:
    """Check for the following error:

    - Map field name not matching any known field

    Examples:
        >>> list(validate_is_map_field_known("a%key.c", {"a%", "a%key.b%"}))
        []
        >>> list(validate_is_map_field_known("a%value.b%", {"a%", "a%value.b%"}))
        []
        >>> list(validate_is_map_field_known("a%key.c%value", {"a%", "a%key.b%", "a%value.b%"}))
        ["Map field 'a%key.c%' does not exist: Did you mean one of the following? [a%key.b%];"]
        >>> list(validate_is_map_field_known("a%key.c", {"b%", "b%key.c%"}))
        ["Map field 'a%' does not exist: Did you mean one of the following? [b%];"]
    """

    def build_message(prefix: str, last_valid_prefix: str) -> str:
        candidates = _find_fields_starting_with_prefix(
            last_valid_prefix, known_map_fields, MAP_MARKER
        )
        return f"Map field '{prefix}' does not exist: Did you mean one of the following? [{', '.join(candidates)}];"

    prefix = ""
    for part in field_name.split(MAP_MARKER)[:-1]:
        last_valid_prefix = prefix
        if part.startswith(MAP_KEY):
            last_valid_prefix += MAP_KEY
        if part.startswith(MAP_VALUE):
            last_valid_prefix += MAP_VALUE
        prefix += part + MAP_MARKER
        if prefix not in known_map_fields:
            yield build_message(prefix, last_valid_prefix)
            return


def validate_nested_field_names(
    *field_names: str,
    allow_maps: bool = True,
    known_fields: Optional[List[str]] = None,
) -> None:
    """Perform various checks on the given nested field names and raise an `spark_frame.utils.AnalysisException`
    if any is found.

    Possible errors:
    - Repeated field marker `!` followed by something else than a struct `.` separator
    - Map marker `%` followed by something else than `%key` or `%value`
    - Map field marker `%` used when maps are not allowed
    - Repeated field name not matching any known field

    Args:
        field_names: A list of nested field names
        allow_maps: Set to true if the transformation calling this method supports maps
        known_fields: List of field names already known

    Raises:
        spark_frame.utils.AnalysisException: if any error is found
    """

    def fail_if_errors(errors: List[str]) -> None:
        if len(errors) > 0:
            raise AnalysisException(errors[0])

    def iterate() -> Generator[str, None, None]:
        for field_name in field_names:
            yield from validate_field_marker_followed_by_non_struct_character(
                field_name
            )
            if allow_maps:
                yield from validate_map_marker_followed_by_non_key_value(field_name)
            else:
                yield from validate_no_map(field_name)
            if known_fields is not None:
                known_repeated_fields = _get_repeated_fields(known_fields)
                yield from validate_is_repeated_field_known(
                    field_name, known_repeated_fields
                )
                known_map_fields = _get_map_fields(known_fields)
                yield from validate_is_map_field_known(field_name, known_map_fields)

    fail_if_errors(list(iterate()))


def resolve_nested_fields(
    fields: Mapping[str, AnyKindOfTransformation],
    starting_level: Union[Column, DataFrame, None] = None,
) -> List[Column]:
    """Builds a list of column expressions to manipulate structs and repeated records

    The syntax for field names works as follows:

    - "." is the separator for struct elements
    - "!" must be appended at the end of fields of type ARRAY
    - "%key" and "%value" must be appended at the end of fields of type MAP

    The following types of transformation are allowed:

    - String and column expressions can be used on any non-repeated field, even nested ones.
    - When working on repeated fields, transformations must be expressed as higher order functions
      (e.g. lambda expressions)
    - `None` can also be used to represent the identity transformation, this is useful to select a field without
       changing and without having to repeat its name.

    Args:
        fields: A mapping (field_name -> transformation to apply to this field)
        starting_level: Nesting level from which the resolution should be started.

    Returns:
        A list of Columns that can be passed to `DataFrame.select` to apply the corresponding transformation to each
        nested column

    Examples:
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
        >>> res_df = df.select(*resolve_nested_fields({"e!!": lambda e: e.cast("DOUBLE")}))
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
    """
    if isinstance(starting_level, DataFrame):
        from spark_frame import nested

        known_fields = nested.fields(starting_level)
    else:
        known_fields = None
    validate_nested_field_names(*fields.keys(), known_fields=known_fields)
    tree = _build_nested_struct_tree(fields)
    root_transformation = _build_transformation_from_tree(tree)
    return root_transformation([starting_level])


def _get_deepest_unnested_field(col_names: List[str]) -> str:
    """Given a list of field names, give the name of the deepest that has been unnested in that list.

    >>> _get_deepest_unnested_field(['id1', 'id2'])
    ''
    >>> _get_deepest_unnested_field(['id1', 'id2', 's1!.id'])
    's1'
    >>> _get_deepest_unnested_field(['id1', 'id2', 's1!.id', 's1!.ss!'])
    's1!.ss'

    """
    deepest = sorted(col_names, key=lambda s: s.count(REPETITION_MARKER))[-1]
    if deepest.count(REPETITION_MARKER) == 0:
        return ""
    else:
        return substring_before_last_occurrence(deepest, REPETITION_MARKER)


def unnest_fields(
    df: DataFrame,
    fields: Union[str, List[str]],
    keep_fields: Optional[List[str]] = None,
) -> Dict[str, DataFrame]:
    """Given a DataFrame, return a list of DataFrames where all the specified columns have been recursively
    unnested (a.k.a. exploded). This produce one DataFrame for each possible granularity.

    !!! warning "Limitation: Maps are not unnested"
        - Fields of type Maps are not unnested by this method.
        - A possible workaround is to first use the transformation
        [`spark_frame.transformations.convert_all_maps_to_arrays`]
        [spark_frame.transformations_impl.convert_all_maps_to_arrays.convert_all_maps_to_arrays]

    Args:
        df: A Spark DataFrame
        fields: One or several nested field names.
        keep_fields: Optional list of field names that should be kept while exploding the DataFrame.

    Returns:
        A list of DataFrames

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql import functions as f
        >>> from spark_frame import nested
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''SELECT
        ...     1 as id,
        ...     ARRAY(STRUCT(2 as a, ARRAY(STRUCT(3 as c, 4 as d)) as b, ARRAY(5, 6) as e)) as s1,
        ...     STRUCT(7 as f) as s2,
        ...     ARRAY(ARRAY(1, 2), ARRAY(3, 4)) as s3,
        ...     ARRAY(ARRAY(STRUCT(1 as e, 2 as f)), ARRAY(STRUCT(3 as e, 4 as f))) as s4
        ... ''')
        >>> nested.fields(df)
        ['id', 's1!.a', 's1!.b!.c', 's1!.b!.d', 's1!.e!', 's2.f', 's3!!', 's4!!.e', 's4!!.f']
        >>> df.show(truncate=False)
        +---+-----------------------+---+----------------+--------------------+
        |id |s1                     |s2 |s3              |s4                  |
        +---+-----------------------+---+----------------+--------------------+
        |1  |[{2, [{3, 4}], [5, 6]}]|{7}|[[1, 2], [3, 4]]|[[{1, 2}], [{3, 4}]]|
        +---+-----------------------+---+----------------+--------------------+
        <BLANKLINE>
        >>> for cols, res_df in unnest_fields(df, ['id', 's2.f']).items():
        ...     print(cols)
        ...     res_df.show()
        <BLANKLINE>
        +---+----+
        | id|s2.f|
        +---+----+
        |  1|   7|
        +---+----+
        <BLANKLINE>
        >>> for cols, res_df in unnest_fields(df, 's1!').items():
        ...     print(cols)
        ...     res_df.show(truncate=False)
        s1
        +---------------------+
        |s1!                  |
        +---------------------+
        |{2, [{3, 4}], [5, 6]}|
        +---------------------+
        <BLANKLINE>
        >>> for cols, res_df in unnest_fields(df, 's1!.b!').items():
        ...     print(cols)
        ...     res_df.show(truncate=False)
        s1!.b
        +------+
        |s1!.b!|
        +------+
        |{3, 4}|
        +------+
        <BLANKLINE>
        >>> for cols, res_df in unnest_fields(df, 's1!.e!').items():
        ...     print(cols)
        ...     res_df.show(truncate=False)
        s1!.e
        +------+
        |s1!.e!|
        +------+
        |5     |
        |6     |
        +------+
        <BLANKLINE>
        >>> for cols, res_df in unnest_fields(df, 's1!.e').items():
        ...     print(cols)
        ...     res_df.show(truncate=False)
        s1
        +------+
        |s1!.e |
        +------+
        |[5, 6]|
        +------+
        <BLANKLINE>
        >>> for cols, res_df in unnest_fields(df, ['s1!.b','s1!.e']).items():
        ...     print(cols)
        ...     res_df.show(truncate=False)
        s1
        +--------+------+
        |s1!.b   |s1!.e |
        +--------+------+
        |[{3, 4}]|[5, 6]|
        +--------+------+
        <BLANKLINE>
        >>> for cols, res_df in unnest_fields(df, 's3!').items():
        ...     print(cols)
        ...     res_df.show(truncate=False)
        s3
        +------+
        |s3!   |
        +------+
        |[1, 2]|
        |[3, 4]|
        +------+
        <BLANKLINE>
        >>> for cols, res_df in unnest_fields(df, 's3!!').items():
        ...     print(cols)
        ...     res_df.show(truncate=False)
        s3!
        +----+
        |s3!!|
        +----+
        |1   |
        |2   |
        |3   |
        |4   |
        +----+
        <BLANKLINE>

        >>> from spark_frame import nested
        >>> for cols, res_df in unnest_fields(df, nested.fields(df), keep_fields=["id"]).items():
        ...     print(cols)
        ...     res_df.show(truncate=False)
        <BLANKLINE>
        +---+----+
        |id |s2.f|
        +---+----+
        |1  |7   |
        +---+----+
        <BLANKLINE>
        s1
        +---+-----+
        |id |s1!.a|
        +---+-----+
        |1  |2    |
        +---+-----+
        <BLANKLINE>
        s1!.b
        +---+--------+--------+
        |id |s1!.b!.c|s1!.b!.d|
        +---+--------+--------+
        |1  |3       |4       |
        +---+--------+--------+
        <BLANKLINE>
        s1!.e
        +---+------+
        |id |s1!.e!|
        +---+------+
        |1  |5     |
        |1  |6     |
        +---+------+
        <BLANKLINE>
        s3!
        +---+----+
        |id |s3!!|
        +---+----+
        |1  |1   |
        |1  |2   |
        |1  |3   |
        |1  |4   |
        +---+----+
        <BLANKLINE>
        s4!
        +---+------+------+
        |id |s4!!.e|s4!!.f|
        +---+------+------+
        |1  |1     |2     |
        |1  |3     |4     |
        +---+------+------+
        <BLANKLINE>

        Making sure keep_columns works with columns inside structs
        >>> for cols, res_df in unnest_fields(df, 's1!', keep_fields=["s2.f"]).items():
        ...     print(cols)
        ...     res_df.show(truncate=False)
        s1
        +----+---------------------+
        |s2.f|s1!                  |
        +----+---------------------+
        |7   |{2, [{3, 4}], [5, 6]}|
        +----+---------------------+
        <BLANKLINE>

        Making sure keep_columns works with columns inside arrays of structs
        >>> for cols, res_df in unnest_fields(df, ['s1!.b!.c', 's1!.b!.d'], keep_fields=["s1!.a"]).items():
        ...     print(cols)
        ...     res_df.show(truncate=False)
        s1!.b
        +-----+--------+--------+
        |s1!.a|s1!.b!.c|s1!.b!.d|
        +-----+--------+--------+
        |2    |3       |4       |
        +-----+--------+--------+
        <BLANKLINE>

    """
    if keep_fields is None:
        keep_columns_list = []
    else:
        keep_columns_list = keep_fields
    if isinstance(fields, str):
        fields = [fields]

    def recurse_node_with_multiple_items(
        node: OrderedTree,
        current_df: DataFrame,
        prefix: str,
        quoted_prefix: str,
    ) -> Generator[Tuple[DataFrame, Column], None, None]:
        for key, children in node.items():
            yield from recurse_item(
                node, key, children, current_df, prefix, quoted_prefix
            )

    def recurse_node_with_one_item(
        children: Optional[OrderedTree],
        current_df: DataFrame,
        prefix: str,
        quoted_prefix: str,
    ) -> Generator[Tuple[DataFrame, Column], None, None]:
        has_children = children is not None
        if has_children:
            node = cast(OrderedTree, children)
            assert_true(
                len(node) == 1,
                "Error, this should not happen: non-struct node with more than one child",
            )
            yield from recurse_node_with_multiple_items(
                node, current_df, prefix=prefix, quoted_prefix=quoted_prefix
            )
        else:
            yield current_df, f.col(quoted_prefix).alias(prefix)

    def recurse_item(
        node: OrderedTree,
        key: str,
        children: Optional[OrderedTree],
        current_df: DataFrame,
        prefix: str,
        quoted_prefix: str,
    ) -> Generator[Tuple[DataFrame, Column], None, None]:
        if key == STRUCT_SEPARATOR:
            assert_true(
                len(node) == 1,
                "Error, this should not happen: tree node of type struct with siblings",
            )
            has_children = children is not None
            assert_true(
                has_children, "Error, this should not happen: struct without children"
            )
            yield from recurse_node_with_multiple_items(
                children,
                current_df,
                prefix=prefix + key,
                quoted_prefix=quoted_prefix + key,
            )
        elif key == REPETITION_MARKER:
            assert_true(
                len(node) == 1,
                "Error, this should not happen: tree node of type array with siblings",
            )
            exploded_col = f.explode(f.col(quoted_prefix)).alias(prefix + key)

            keep_cols = [
                f.col(keep_col).alias(keep_col)
                for keep_col in keep_columns_list
                if is_direct_sub_field_of_any(keep_col, current_df.columns)
                or keep_col in current_df.columns
            ]
            new_df = current_df.select(*keep_cols, exploded_col)
            yield from recurse_node_with_one_item(
                children,
                new_df,
                prefix=prefix + key,
                quoted_prefix=quote(prefix + key),
            )
        else:
            yield from recurse_node_with_one_item(
                children,
                current_df,
                prefix=prefix + key,
                quoted_prefix=quoted_prefix + quote(key),
            )

    col_dict = {col: None for col in fields}
    root_tree = _build_nested_struct_tree(col_dict)
    dataframe_and_columns = recurse_node_with_multiple_items(
        root_tree, df, prefix="", quoted_prefix=""
    )
    grouped_res = group_by_key(dataframe_and_columns)
    res = [
        df.select(
            *[
                quote(keep_col)
                for keep_col in keep_columns_list
                if keep_col in df.columns and keep_col not in df.select(*cols).columns
            ],
            *cols,
        )
        for df, cols in grouped_res.items()
    ]
    return {_get_deepest_unnested_field(df.columns): df for df in res}
