import re
from typing import List, Optional, Union, cast

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import DataType, StructField, StructType

StringOrColumn = Union[str, Column]


def quote(col: str) -> str:
    """Puts the given column name into quotes.

    This is useful in particular when some column names contains dots,
    which usually happens after using the :func:`flatten` operation.

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

    :param col:
    :return:
    """
    return "`" + col.replace("`", "``") + "`"


def quote_idempotent(col: str) -> str:
    """Puts the given column name into quotes if it is not already

    This is useful in particular when some column names contains dots,
    which usually happens after using the :func:`flatten` operation.

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

    :param col:
    :return:
    """
    return quote(unquote(col))


def quote_columns(columns: List[str]) -> List[str]:
    """Puts every column name of the given list into quotes.

    This is useful in particular when some column names contains dots,
    which usually happens after using the :func:`flatten` operation.

    :param columns:
    :return:
    """
    return [quote(col) for col in columns]


def unquote(col_name: str) -> str:
    """Removes quotes from a quoted column name.

    Examples:

    >>> unquote('a')
    'a'
    >>> unquote('`a`')
    'a'
    >>> unquote('`a.b`')
    'a.b'

    Columns names can even contain backquotes, by escaping backquotes with another backquote.

    >>> unquote('`a``b`')
    'a`b'

    :param col_name:
    :return:
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

    :param col_name:
    :return:
    """
    col_parts = [unquote(match[0]) for match in __split_col_name_regex.findall(col_name)]
    return col_parts


def str_to_col(args: StringOrColumn) -> Column:
    """Converts string or Column argument to Column types

    Requires the SparkSession to be instantiated.

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
    if isinstance(args, str):
        return f.col(args)
    else:
        return args


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


def get_nested_col_type_from_schema(col_name: str, schema: StructType) -> DataType:
    """Fetch recursively the DataType of a column inside a DataFrame schema (or more generally any StructType)

    Example:
    >>> from pyspark.sql import SparkSession
    >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
    >>> df = spark.createDataFrame([
    ...      (1, {"a.b" : {"c.d": 1, "e": "1", "f`g": True}}),
    ...      (2, {"a.b" : {"c.d": 2, "e": "2", "f`g": True}}),
    ...      (3, {"a.b" : {"c.d": 3, "e": "3", "f`g": True}}),
    ...   ],
    ...   "id INT, `a.b` STRUCT<`c.d`:INT, e:STRING, `f``g`:BOOLEAN>"
    ... )
    >>> get_nested_col_type_from_schema("`a.b`.`c.d`",df.schema).simpleString()
    'int'
    >>> get_nested_col_type_from_schema("`a.b`.e",df.schema).simpleString()
    'string'
    >>> get_nested_col_type_from_schema("`a.b`.`f``g`",df.schema).simpleString()
    'boolean'

    :param schema: the DataFrame schema (or StructField) in which the column type will be fetched.
    :param col_name: the name of the column to get
    :return:
    """
    col_parts = split_col_name(col_name)

    def get_col(col: str, fields: List[StructField]) -> StructField:
        for field in fields:
            if field.name == col:
                return field
        raise ValueError(f'Cannot resolve column name "{col_name}"')

    struct: Union[StructType, DataType] = schema
    for col_part in col_parts:
        assert_true(isinstance(struct, StructType))
        struct = cast(StructType, struct)
        struct = get_col(col_part, struct.fields).dataType
    assert_true(isinstance(struct, DataType))
    return cast(DataType, struct)


def get_instantiated_spark_session() -> SparkSession:
    """Get the instantiated SparkSession. Raises an AssertionError if it does not exists."""
    optional_spark = getattr(SparkSession, "_instantiatedSession")
    assert_true(optional_spark is not None)
    spark = cast(SparkSession, optional_spark)
    return spark


def show_string(df: DataFrame, n: int = 20, truncate: Union[bool, int] = True, vertical: bool = False) -> str:
    """Write the first ``n`` rows to the console into a string.
    This is similar to [DataFrame.show](pyspark.sql.DataFrame.show) except it returns a string instead of directly
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
    This is similar to [DataFrame.printSchema](pyspark.sql.DataFrame.printSchema) except it returns a string instead
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


def safe_struct_get(s: Optional[Column], field: str) -> Column:
    """Get a column's subfield, unless the column is None, in which case it returns
    a Column expression for the field itself."""
    if s is None:
        return f.col(field)
    else:
        return s[field]


def assert_true(assertion: bool, error: Union[str, BaseException] = None) -> None:
    """Raise an Exception with the given error_message if the assertion passed is false.

    !!! Tip
        This method is especially useful to get 100% coverage more easily, without having to write tests for every
        single assertion to cover the cases when they fail (which are generally just there to provide a more helpful
        error message to users when something that is not supposed to happen does happen)

    Args:
        assertion: The boolean result of an assertion
        error: An Exception or a message string (in which case an AssertError with this message will be raised)

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
