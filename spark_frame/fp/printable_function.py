from typing import Any, Callable, Optional, Union, cast


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

    def __init__(self, func: Callable[[Any], Any], alias: Union[str, Callable[[Optional[str]], str]]) -> None:
        self.func: Callable[[Any], Any] = func
        self.alias: Union[str, Callable[[Optional[str]], str]] = alias

    def __repr__(self) -> str:
        if callable(self.alias):
            return f"lambda x: {cast(Callable, self.alias)('x')}"
        else:
            return cast(str, self.alias)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def apply_alias(self, x: Optional[str]) -> str:
        if callable(self.alias):
            return self.alias(x)
        else:
            return self.alias
