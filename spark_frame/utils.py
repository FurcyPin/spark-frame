import re
from typing import List, Union, cast

from pyspark.sql import SparkSession
from pyspark.sql.types import DataType, StructField, StructType


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


def assert_true(assertion: bool, error_message: str = None) -> None:
    """Raise a ValueError with the given error_message if the assertion passed is false

    >>> assert_true(3==4, "3 <> 4")
    Traceback (most recent call last):
    ...
    AssertionError: 3 <> 4

    >>> assert_true(3==3, "3 <> 4")

    :param assertion: assertion that will be checked
    :param error_message: error message to display if the assertion is false
    """
    if not assertion:
        if error_message is None:
            raise AssertionError()
        else:
            raise AssertionError(error_message)
