from typing import Callable, cast

from spark_frame.fp.printable_function import PrintableFunction


def compose(f1: PrintableFunction, f2: PrintableFunction) -> PrintableFunction:
    """Composes together two PrintableFunctions.
    For instance, if `h = compose(g, f)`, then for every `x`, `h(x) = g(f(x))`.

    Args:
        f1: A PrintableFunction
        f2: A PrintableFunction

    Returns:
        The composition of f1 with f2.

    Examples:

        >>> f = PrintableFunction(lambda x: x+1, lambda s: f'{s} + 1')
        >>> g = PrintableFunction(lambda x: x.cast("Double"), lambda s: f'({s}).cast("Double")')
        >>> compose(g, f)
        lambda x: (x + 1).cast("Double")
        >>> compose(f, g)
        lambda x: (x).cast("Double") + 1

        >>> h = PrintableFunction(lambda x: x*2, "h")
        >>> compose(h, f)
        lambda x: h(x + 1)
        >>> compose(f, h)
        h + 1
        >>> compose(h, h)
        h(h)

    """
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
