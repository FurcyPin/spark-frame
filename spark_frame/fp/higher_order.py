from typing import Any, Callable, List, Optional

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from spark_frame import fp
from spark_frame.fp.printable_function import PrintableFunction
from spark_frame.utils import quote
from spark_frame.utils import str_to_col as _str_to_col


def alias(name: str) -> PrintableFunction:
    """Return a PrintableFunction version of the `pyspark.sql.Column.alias` method"""
    return PrintableFunction(lambda s: s.alias(name), lambda s: str(s) + f".alias({repr(name)})")


identity = PrintableFunction(lambda s: s, lambda s: str(s))
struct = PrintableFunction(lambda x: f.struct(x), lambda x: f"f.struct({x})")
str_to_col = PrintableFunction(lambda x: _str_to_col(x), lambda s: str(s))


def struct_get(key: str) -> PrintableFunction:
    """Return a PrintableFunction that gets a struct's subfield, unless the struct is None,
    in which case it returns a Column expression for the field itself.

    Get a column's subfield, unless the column is None, in which case it returns
    a Column expression for the field itself.

    Examples:
        >>> struct_get("c")
        lambda x: x['c']
        >>> struct_get("c").alias(None)
        "f.col('c')"
    """

    def _safe_struct_get(s: Optional[Column], field: str) -> Column:
        if s is None:
            return f.col(field)
        else:
            if ("." in field or "!" in field) and isinstance(s, DataFrame):
                return s[quote(field)]
            else:
                return s[field]

    def _safe_struct_get_alias(s: Optional[str], field: str) -> str:
        if s is None:
            return f"f.col({repr(field)})"
        else:
            return f"{s}[{repr(field)}]"

    return PrintableFunction(lambda s: _safe_struct_get(s, key), lambda s: _safe_struct_get_alias(s, key))


def recursive_struct_get(keys: List[str]) -> PrintableFunction:
    """Return a PrintableFunction that recursively applies get to a nested structure.

    Examples:
        >>> recursive_struct_get([])
        lambda x: x
        >>> recursive_struct_get(["a", "b", "c"])
        lambda x: x['a']['b']['c']
        >>> recursive_struct_get(["a", "b", "c"]).alias(None)
        "f.col('a')['b']['c']"
    """
    if len(keys) == 0:
        return identity
    else:
        return fp.compose(recursive_struct_get(keys[1:]), struct_get(keys[0]))


def transform(transformation: PrintableFunction) -> PrintableFunction:
    """Return a PrintableFunction version of the `pyspark.sql.functions.transform` method,
    which applies the given transformation to any array column.
    """
    return PrintableFunction(
        lambda x: f.transform(x, transformation.func),
        lambda x: f"f.transform({x}, lambda x: {transformation.alias('x')})",
    )


def transform_keys(transformation: PrintableFunction) -> PrintableFunction:
    """Return a PrintableFunction version of the `pyspark.sql.functions.transform_keys` method,
    which applies the given transformation to any array column.
    """
    return PrintableFunction(
        lambda x: f.transform_keys(x, lambda k, v: transformation.func(k)),
        lambda x: f"f.transform_keys({x}, lambda k, v: {transformation.alias('k')})",
    )


def transform_values(transformation: PrintableFunction) -> PrintableFunction:
    """Return a PrintableFunction version of the `pyspark.sql.functions.transform_values` method,
    which applies the given transformation to any array column.
    """
    return PrintableFunction(
        lambda x: f.transform_values(x, lambda k, v: transformation.func(v)),
        lambda x: f"f.transform_values({x}, lambda k, v: {transformation.alias('v')})",
    )


def _partial_box_right(func: Callable, args: Any) -> Callable:
    """Given a function and an array of arguments, return a new function that takes an argument, add it to the
    array, and pass it to the original function."""
    if isinstance(args, str):
        args = [args]
    return lambda a: func(args + [a])


def boxed_transform(transformation: PrintableFunction, parents: List[str]) -> PrintableFunction:
    """Return a PrintableFunction version of the `pyspark.sql.functions.transform` method,
    which applies the given transformation to any array column.
    """
    return PrintableFunction(
        lambda x: f.transform(recursive_struct_get(parents)(x[-1]), _partial_box_right(transformation.func, x)),
        lambda x: f"f.transform({recursive_struct_get(parents).alias(x[-1])}, "
        f"lambda x: {_partial_box_right(transformation.alias, x)('x')})",
    )


def boxed_transform_map(
    key_transformation: PrintableFunction, value_transformation: PrintableFunction, parents: List[str]
) -> PrintableFunction:
    """Return a PrintableFunction version of the `pyspark.sql.functions.transform_keys` method,
    which applies the given transformation to any array column.
    """
    return PrintableFunction(
        lambda x: f.transform_values(
            f.transform_keys(
                recursive_struct_get(parents)(x[-1]),
                lambda k, v: _partial_box_right(key_transformation.func, x)(k),
            ),
            lambda k, v: _partial_box_right(value_transformation.func, x)(v),
        ),
        lambda x: "f.transform_values("
        "f.transform_keys("
        f"{recursive_struct_get(parents).alias(x[-1])}, "
        f"lambda k, v: {_partial_box_right(key_transformation.alias, x)('k')})"
        "),"
        f"lambda k, v: {_partial_box_right(value_transformation.alias, x)('v')}"
        ")",
    )
