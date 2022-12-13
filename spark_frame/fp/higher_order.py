from spark_frame.fp.printable_function import PrintableFunction

identity = PrintableFunction(lambda s: s, lambda s: s)


def __safe_struct_get_alias(s: str, field: str) -> str:
    """Return a string representation for the `safe_struct_get` method."""
    if s is None:
        return f"f.col('{field}')"
    else:
        return f"{s}[{field}]"


def safe_struct_get(key: str) -> PrintableFunction:
    """Return a PrintableFunction of the `spark_frame.utils.safe_struct_get` method:

    Get a column's subfield, unless the column is None, in which case it returns
    a Column expression for the field itself.
    """
    from spark_frame.utils import safe_struct_get as _safe_struct_get

    return PrintableFunction(lambda s: _safe_struct_get(s, key), lambda s: __safe_struct_get_alias(s, key))
