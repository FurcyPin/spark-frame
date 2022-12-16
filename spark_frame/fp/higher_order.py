from typing import Callable, Optional, Union

from pyspark.sql import functions as f

from spark_frame.fp.printable_function import PrintableFunction


def __get_func(function: Union[Callable, PrintableFunction]) -> Callable:
    """Given a ColumnTransformation, printable or not, return a ColumnTransformation"""
    if isinstance(function, PrintableFunction):
        return function.func
    else:
        return function


def __safe_struct_get_alias(s: Optional[str], field: str) -> str:
    """Return a string representation for the `safe_struct_get` method."""
    if s is None:
        return f"f.col('{field}')"
    else:
        return f"{s}[{field}]"


def alias(name: str) -> PrintableFunction:
    """Return a PrintableFunction version of the `pyspark.sql.Column.alias` method"""
    return PrintableFunction(lambda s: s.alias(name), lambda s: str(s) + f".alias({name})")


identity = PrintableFunction(lambda s: s, lambda s: str(s))
sort_array = PrintableFunction(lambda x: f.sort_array(x), lambda x: f"f.sort_array({x})")
struct = PrintableFunction(lambda x: f.struct(x), lambda x: f"f.struct({x})")


def safe_struct_get(key: str) -> PrintableFunction:
    """Return a PrintableFunction version of the `spark_frame.utils.safe_struct_get` method:

    Get a column's subfield, unless the column is None, in which case it returns
    a Column expression for the field itself.

    >>> safe_struct_get("c")
    lambda x: x[c]
    >>> safe_struct_get("c").alias(None)
    "f.col('c')"
    """
    from spark_frame.utils import safe_struct_get as _safe_struct_get

    return PrintableFunction(lambda s: _safe_struct_get(s, key), lambda s: __safe_struct_get_alias(s, key))


def transform(transformation: Union[Callable, PrintableFunction]) -> PrintableFunction:
    """Return a PrintableFunction version of the `pyspark.sql.functions.transform` method, which applies the given
    transformation to any array column.
    """
    return PrintableFunction(
        lambda x: f.transform(x, __get_func(transformation)),
        lambda x: f"f.transform({x}, {transformation})",
    )
