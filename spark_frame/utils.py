import importlib
import re
from typing import Dict, Iterable, List, Tuple, TypeVar, Union, cast

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as f

from spark_frame.conf import PROJECT_NAME

StringOrColumn = Union[str, Column]
K = TypeVar("K")
V = TypeVar("V")

MAX_JAVA_INT = 2147483647


class AnalysisException(Exception):
    pass


def is_sub_field(sub_field: str, field: str):
    """Return True if `sub_field` is a sub-field of `field`

    >>> is_sub_field("a", "a")
    True
    >>> is_sub_field("a", "b")
    False

    >>> is_sub_field("a.b", "a")
    True
    >>> is_sub_field("a.b", "b")
    False

    >>> is_sub_field("a", "a.b")
    False

    """
    return sub_field == field or sub_field.startswith(field + ".")


def is_sub_field_of_any(sub_field: str, fields: List[str]):
    """Return True if `sub_field` is a sub-field of any field in `fields`

    >>> is_sub_field_of_any("a", ["a", "b"])
    True
    >>> is_sub_field_of_any("a", ["b", "c"])
    False

    >>> is_sub_field_of_any("a.b", ["a", "b"])
    True
    >>> is_sub_field_of_any("a.b", ["b", "c"])
    False

    >>> is_sub_field_of_any("a", ["a.b"])
    False

    >>> is_sub_field_of_any("a", [])
    False

    """
    return any([is_sub_field(sub_field, field) for field in fields])


def group_by_key(items: Iterable[Tuple[K, V]]) -> Dict[K, List[V]]:
    """Group the values of a list of tuples by their key.

    Args:
        items: An iterable of tuples (key, value).

    Returns:
        A dictionary where the keys are the keys from the input tuples,
        and the values are lists of the corresponding values.

    Examples:
        >>> items = [('a', 1), ('b', 2), ('a', 3), ('c', 4), ('b', 5)]
        >>> group_by_key(items)
        {'a': [1, 3], 'b': [2, 5], 'c': [4]}
        >>> group_by_key([])
        {}
    """
    result: Dict[K, List[V]] = {}
    for key, value in items:
        if key in result:
            result[key].append(value)
        else:
            result[key] = [value]
    return result


def quote(col: str) -> str:
    """Puts the given column name into quotes.

    This is useful in particular when some column names contains dots,
    which usually happens after using the `flatten` transformation.

    Args:
        col: A column name

    Returns:
        The column name put inside quotes

    Examples:
        >>> quote('a')
        '`a`'
        >>> quote('a.b')
        '`a.b`'

        Column names can even contain backquotes, by escaping backquotes with another backquote.

        >>> quote('a`b')
        '`a``b`'

        Differs from `quote_idempotent` on this specific case
        >>> quote('`a`')
        '```a```'
    """
    return "`" + col.replace("`", "``") + "`"


def quote_idempotent(col: str) -> str:
    """Puts the given column name into quotes if it is not already.

    This is useful in particular when some column names contains dots,
    which usually happens after using the `flatten` transformation.

    Args:
        col: A column name

    Returns:
        The column name put inside quotes if it is not already

    Examples:
        >>> quote_idempotent('a')
        '`a`'
        >>> quote_idempotent('a.b')
        '`a.b`'

        Column names can even contain backquotes, by escaping backquotes with another backquote.

        >>> quote_idempotent('a`b')
        '`a``b`'

        Differs from `quote` on this specific case
        >>> quote_idempotent('`a`')
        '`a`'
    """
    return quote(unquote(col))


def quote_columns(columns: List[str]) -> List[str]:
    """Puts every column name of the given list into quotes.

    This is useful in particular when some column names contains dots,
    which usually happens after using the :func:`flatten` operation.

    Args:
        columns: A list of column names

    Returns:
        A list with each column name put into quotes

    """
    return [quote(col) for col in columns]


def unquote(col_name: str) -> str:
    """Removes quotes from a quoted column name.

    Args:
        col_name: A column name

    Returns:
        The column name with quotes removed

    Examples:
        >>> unquote('a')
        'a'
        >>> unquote('`a`')
        'a'
        >>> unquote('`a.b`')
        'a.b'

        Column names can even contain backquotes, by escaping backquotes with another backquote.

        >>> unquote('`a``b`')
        'a`b'
    """
    if col_name[0] == "`" and col_name[-1] == "`":
        col_name = col_name[1:-1]
    return col_name.replace("``", "`")


__split_col_name_regex = re.compile(
    r"""(                   # Matching group 1 [Everything that comes before the first DOT split]
            (?:                 # Non-matching group
                    [^`.]*          # Everything but BACKQUOTE and DOT zero or more times
                    `               # BACKQUOTE
                    [^`]*           # Everything but BACKQUOTE zero or more times [DOTS are allowed between BACKQUOTES]
                    `               # BACKQUOTE
                |               # OR
                    [^`.]+          # Everything but BACKQUOTE and DOT one or more times
            )+                  # Repeat this group one or more times
        )
        ([.]|$)             # Matching group 2 [The separator: DOT or end of line (to match strings without dots)]
    """,
    re.VERBOSE,  # Allows to add comments inside the regex
)


def split_col_name(col_name: str) -> List[str]:
    """Splits a Spark column name representing a nested field into a list of path parts.

    Args:
        col_name: A column name

    Returns:
        The column name split by nesting level

    Examples:
        In this example: `a` is a struct containing a field `b`.

        >>> split_col_name("a.b")
        ['a', 'b']
        >>> split_col_name("ab")
        ['ab']

        Field names can contain dots when escaped between backquotes (`)

        >>> split_col_name("`a.b`")
        ['a.b']
        >>> split_col_name("`a.b`.`c.d`")
        ['a.b', 'c.d']
        >>> split_col_name("`a.b`.c.`d`")
        ['a.b', 'c', 'd']

        Field names can even contain backquotes, by escaping backquotes with another backquote.

        >>> split_col_name("`a``b`")
        ['a`b']
        >>> split_col_name("`.a.``.b.`")
        ['.a.`.b.']
        >>> split_col_name("`ab`.`c``d`.fg")
        ['ab', 'c`d', 'fg']
    """
    col_parts = [unquote(match[0]) for match in __split_col_name_regex.findall(col_name)]
    return col_parts


def str_to_col(col: StringOrColumn) -> Column:
    """Converts string or Column argument to Column types

    Requires the SparkSession to be instantiated.

    Args:
        col: A column name or Column expression

    Returns:
        A Column expression

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> str_to_col("id")
        Column<'id'>
        >>> str_to_col(f.expr("COUNT(1)"))
        Column<'COUNT(1)'>
        >>> str_to_col("*")
        Column<'unresolvedstar()'>
    """
    if isinstance(col, str):
        return f.col(col)
    else:
        return col


def strip_margin(text: str):
    """For every line in this string, strip a leading prefix consisting of whitespace, tabs and carriage returns
    followed by | from the line.

    If the first character is a newline, it is also removed.
    This method is inspired from Scala's String.stripMargin.

    Args:
        text: A multi-line string

    Returns:
        A stripped string

    Examples:
        >>> print(strip_margin('''
        ...     |a
        ...     |b
        ...     |c'''))
        a
        b
        c
        >>> print(strip_margin('''a
        ... |b
        ...   |c
        ...     |d'''))
        a
        b
        c
        d
    """
    s = re.sub(r"\n[ \t\r]*\|", "\n", text)
    if s.startswith("\n"):
        return s[1:]
    else:
        return s


def get_instantiated_spark_session() -> SparkSession:
    """Get the instantiated SparkSession. Raises an AssertionError if it does not exists."""
    optional_spark = getattr(SparkSession, "_instantiatedSession")
    assert_true(optional_spark is not None)
    spark = cast(SparkSession, optional_spark)
    return spark


def show_string(df: DataFrame, n: int = 20, truncate: Union[bool, int] = True, vertical: bool = False) -> str:
    """Write the first ``n`` rows to the console into a string.
    This is similar to [DataFrame.show][pyspark.sql.DataFrame.show] except it returns a string instead of directly
    writing it to stdout.

    Args:
        df: A Spark DataFrame
        n: Number of rows to show.
        truncate: If set to ``True``, truncate strings longer than 20 chars by default.
            If set to a number greater than one, truncates long strings to length ``truncate``
            and align cells right.
        vertical: If set to ``True``, print output rows vertically (one line per column value).

    Returns:
        A string representing the first `n` rows of the DataFrame

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''
        ...  SELECT INLINE(ARRAY(
        ...    STRUCT(2 as age, "Alice" as name),
        ...    STRUCT(5 as age, "Bob" as name)
        ...  ))
        ... ''')
        >>> df
        DataFrame[age: int, name: string]
        >>> print(show_string(df))
        +---+-----+
        |age| name|
        +---+-----+
        |  2|Alice|
        |  5|  Bob|
        +---+-----+
        <BLANKLINE>
        >>> print(show_string(df, truncate=3))
        +---+----+
        |age|name|
        +---+----+
        |  2| Ali|
        |  5| Bob|
        +---+----+
        <BLANKLINE>
        >>> print(show_string(df, vertical=True))  # doctest: +NORMALIZE_WHITESPACE
        -RECORD 0-----
         age  | 2
         name | Alice
        -RECORD 1-----
         age  | 5
         name | Bob
        <BLANKLINE>
    """
    assert_true(isinstance(n, (int, bool)), TypeError("Parameter 'n' (number of rows) must be an int"))
    assert_true(isinstance(vertical, bool), TypeError("Parameter 'vertical' must be a bool"))
    assert_true(
        isinstance(truncate, (int, bool)), TypeError(f"Parameter 'truncate={truncate}' should be either bool or int.")
    )

    if isinstance(truncate, bool) and truncate:
        return df._jdf.showString(n, 20, vertical)
    else:
        return df._jdf.showString(n, int(truncate), vertical)


def schema_string(df: DataFrame) -> str:
    """Write the DataFrame schema to a string.
    This is similar to [DataFrame.printSchema][pyspark.sql.DataFrame.printSchema] except it returns a string instead
    of directly writing it to stdout.

    Args:
        df: A Spark DataFrame

    Returns:
        a string representing the schema as a tree

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''
        ...  SELECT INLINE(ARRAY(
        ...    STRUCT(2 as age, "Alice" as name),
        ...    STRUCT(5 as age, "Bob" as name)
        ...  ))
        ... ''')
        >>> print(schema_string(df))
        root
         |-- age: integer (nullable = false)
         |-- name: string (nullable = false)
        <BLANKLINE>
    """
    return df._jdf.schema().treeString()


def assert_true(assertion: bool, error: Union[str, BaseException] = None) -> None:
    """Raise an Exception with the given error_message if the assertion passed is false.

    !!! Tip
        This method is especially useful to get 100% coverage more easily, without having to write tests for every
        single assertion to cover the cases when they fail (which are generally just there to provide a more helpful
        error message to users when something that is not supposed to happen does happen)

    Args:
        assertion: The boolean result of an assertion
        error: An Exception or a message string (in which case an AssertError with this message will be raised)

    Examples:
        >>> assert_true(3==3, "3 <> 4")
        >>> assert_true(3==4, "3 <> 4")
        Traceback (most recent call last):
        ...
        AssertionError: 3 <> 4
        >>> assert_true(3==4, ValueError("3 <> 4"))
        Traceback (most recent call last):
        ...
        ValueError: 3 <> 4
        >>> assert_true(3==4)
        Traceback (most recent call last):
        ...
        AssertionError
    """
    if not assertion:
        if isinstance(error, BaseException):
            raise error
        elif isinstance(error, str):
            raise AssertionError(error)
        else:
            raise AssertionError()


def load_external_module(module_name: str):
    """Load and return a Python module, raising an exception if it is not installed or does not meet the
    expected version requirements.

    Args:
        module_name: The name of the module to load.

    Returns:
        module: The loaded Python module.

    Raises:
        ImportError: If the module is not found, does not have a `__version__` attribute, or does not meet
            the expected version requirements.

    Examples:
        >>> load_external_module('platform').__name__
        'platform'

        >>> load_external_module('nonexistent_module')
        Traceback (most recent call last):
            ...
        ImportError: Module 'nonexistent_module' not found.
        To keep spark-frame as light, flexible and secure as possible, it was not included in its dependencies.
        Please add it to your project dependencies to use this method.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"Module '{module_name}' not found.\n"
            f"To keep {PROJECT_NAME} as light, flexible and secure as possible, "
            "it was not included in its dependencies.\n"
            "Please add it to your project dependencies to use this method."
        )

    return module
