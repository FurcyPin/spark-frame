import difflib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DataType, MapType, StructField, StructType

from spark_frame.conf import REPETITION_MARKER
from spark_frame.data_type_utils import flatten_schema, is_nullable
from spark_frame.field_utils import has_same_granularity_as_any, is_parent_field_of_any


class DiffPrefix(str, Enum):
    ADDED = "+"
    REMOVED = "-"
    UNCHANGED = " "

    def __repr__(self) -> str:
        return f"'{self.value}'"


@dataclass
class SchemaDiffResult:
    same_schema: bool
    diff_str: str
    nb_cols: int
    column_names_diff: Dict[str, DiffPrefix]
    """The diff per column names.
    Used to determine which columns appeared or disappeared and the order in which the columns shall be displayed"""

    def display(self) -> None:
        if not self.same_schema:
            print(f"Schema has changed:\n{self.diff_str}")
            print("WARNING: columns that do not match both sides will be ignored")
        else:
            print(f"Schema: ok ({self.nb_cols})")

    @property
    def column_names(self) -> List[str]:
        return list(self.column_names_diff.keys())


def _schema_to_string(schema: StructType, include_nullable: bool = False, include_metadata: bool = False) -> List[str]:
    """Return a list of strings representing the schema

    Args:
        schema: A DataFrame's schema
        include_nullable: Indicate for each field if it is nullable
        include_metadata: Add field description

    Returns:

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql.types import IntegerType, StringType, StructField, StructType
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> df = spark.sql('''SELECT 1 as id, "a" as c1, 1 as c2''')
        >>> print('\\n'.join(_schema_to_string(df.schema)))
        id INT
        c1 STRING
        c2 INT
        >>> print('\\n'.join(_schema_to_string(df.schema, include_nullable=True)))
        id INT (not nullable)
        c1 STRING (not nullable)
        c2 INT (not nullable)
        >>> schema = StructType([
        ...     StructField('id', IntegerType(), nullable=True, metadata={"description": 'An id'}),
        ...     StructField('c1', StringType(), nullable=False, metadata={"description": 'A string column'}),
        ...     StructField('c2', IntegerType(), nullable=True, metadata={"description": 'An int column'})
        ... ])
        >>> print('\\n'.join(_schema_to_string(schema, include_nullable=True, include_metadata=True)))
        id INT (nullable) {'description': 'An id'}
        c1 STRING (not nullable) {'description': 'A string column'}
        c2 INT (nullable) {'description': 'An int column'}
        >>> df = spark.sql('''SELECT 1 as id, STRUCT(2 as a, ARRAY(STRUCT(3 as c, 4 as d, ARRAY(5) as e)) as b) as s''')
        >>> print('\\n'.join(_schema_to_string(df.schema)))
        id INT
        s STRUCT<a:INT,b:ARRAY<STRUCT<c:INT,d:INT,e:ARRAY<INT>>>>
        >>> df = spark.sql('''SELECT 1 as id, MAP(1, "a") as m''')
        >>> print('\\n'.join(_schema_to_string(df.schema)))
        id INT
        m MAP<INT,STRING>
    """

    def type_to_string(data_type: DataType) -> str:
        if isinstance(data_type, StructType):
            return f"""STRUCT<{",".join(field_to_string(f) for f in data_type.fields)}>"""
        if isinstance(data_type, ArrayType):
            return f"""ARRAY<{type_to_string(data_type.elementType)}>"""
        if isinstance(data_type, MapType):
            return f"""MAP<{type_to_string(data_type.keyType)},{type_to_string(data_type.valueType)}>"""
        else:
            return f"{data_type.simpleString().upper()}"

    def meta_str(field: StructField) -> str:
        s = ""
        if include_nullable:
            if is_nullable(field):
                s += " (nullable)"
            else:
                s += " (not nullable)"
        if include_metadata:
            s += f" {field.metadata}"
        return s

    def field_to_string(struct_field: StructField, sep: str = ":") -> str:
        return f"{struct_field.name}{sep}{type_to_string(struct_field.dataType)}"

    return [field_to_string(field, sep=" ") + meta_str(field) for field in schema]


def diff_dataframe_schemas(left_df: DataFrame, right_df: DataFrame, join_cols: List[str]) -> SchemaDiffResult:
    """Compares two DataFrames schemas and print out the differences.
    Ignore the nullable and comment attributes.

    Args:
        left_df: A Spark DataFrame
        right_df: Another DataFrame
        join_cols: The list of column names that will be used for joining the two DataFrames together

    Returns:
        A SchemaDiffResult object

    Examples:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> left_df = spark.sql('''SELECT 1 as id, "" as c1, "" as c2, ARRAY(STRUCT(2 as a, "" as b)) as c4''')
        >>> right_df = spark.sql('''SELECT 1 as id, 2 as c1, "" as c3, ARRAY(STRUCT(3 as a, "" as d)) as c4''')
        >>> schema_diff_result = diff_dataframe_schemas(left_df, right_df, ["id"])
        >>> schema_diff_result.display()
        Schema has changed:
        @@ -1,4 +1,4 @@
        <BLANKLINE>
         id INT
        -c1 STRING
        -c2 STRING
        -c4 ARRAY<STRUCT<a:INT,b:STRING>>
        +c1 INT
        +c3 STRING
        +c4 ARRAY<STRUCT<a:INT,d:STRING>>
        WARNING: columns that do not match both sides will be ignored
        >>> schema_diff_result.same_schema
        False
        >>> schema_diff_result.column_names_diff
        {'id': ' ', 'c1': ' ', 'c2': '-', 'c3': '+', 'c4': ' '}

        >>> schema_diff_result = diff_dataframe_schemas(left_df, right_df, ["id", "c4!.a"])
        >>> schema_diff_result.display()
        Schema has changed:
        @@ -1,5 +1,5 @@
        <BLANKLINE>
         id INT
        -c1 STRING
        -c2 STRING
        +c1 INT
        +c3 STRING
         c4!.a INT
        -c4!.b STRING
        +c4!.d STRING
        WARNING: columns that do not match both sides will be ignored
        >>> schema_diff_result.same_schema
        False
        >>> schema_diff_result.column_names_diff
        {'id': ' ', 'c1': ' ', 'c2': '-', 'c3': '+', 'c4!.a': ' ', 'c4!.b': '-', 'c4!.d': '+'}


    """

    def explode_schema_according_to_join_cols(schema: StructType) -> StructType:
        exploded_schema = flatten_schema(schema, explode=True, keep_non_leaf_fields=True)
        return StructType(
            [
                field
                for field in exploded_schema.fields
                if has_same_granularity_as_any(field.name, join_cols)
                and not is_parent_field_of_any(field.name, join_cols)
                and not isinstance(field.dataType, StructType)
                and not field.name.endswith(REPETITION_MARKER)
            ],
        )

    left_schema_flat_exploded = explode_schema_according_to_join_cols(left_df.schema)
    right_schema_flat_exploded = explode_schema_according_to_join_cols(right_df.schema)

    left_schema: List[str] = _schema_to_string(left_schema_flat_exploded)
    right_schema: List[str] = _schema_to_string(right_schema_flat_exploded)
    left_columns_flat: List[str] = [field.name for field in left_schema_flat_exploded.fields]
    right_columns_flat: List[str] = [field.name for field in right_schema_flat_exploded.fields]

    diff_str = list(difflib.unified_diff(left_schema, right_schema, n=10000))[2:]
    column_names_diff = _diff_dataframe_column_names(left_columns_flat, right_columns_flat)
    same_schema = len(diff_str) == 0
    if same_schema:
        diff_str = left_schema
    return SchemaDiffResult(
        same_schema=same_schema,
        diff_str="\n".join(diff_str),
        nb_cols=len(left_df.columns),
        column_names_diff=column_names_diff,
    )


def _remove_potential_duplicates_from_diff(diff: List[str]) -> List[str]:
    """In some cases (e.g. swapping the order of two columns), the difflib.unified_diff produces results
    where a column is added and then removed. This method replaces such duplicates with a single occurrence
    of the column marked as unchanged. We keep the column ordering of the left side.

    Examples:
        >>> _remove_potential_duplicates_from_diff([' id', ' col1', '+col4', '+col3', ' col2', '-col3', '-col4'])
        [' id', ' col1', ' col2', ' col3', ' col4']

    """
    plus = {row[1:] for row in diff if row[0] == DiffPrefix.ADDED}
    minus = {row[1:] for row in diff if row[0] == DiffPrefix.REMOVED}
    both = plus.intersection(minus)
    return [
        DiffPrefix.UNCHANGED + row[1:] if row[1:] in both else row
        for row in diff
        if (row[1:] not in both) or (row[0] == DiffPrefix.REMOVED)
    ]


def _diff_dataframe_column_names(left_col_names: List[str], right_col_names: List[str]) -> Dict[str, DiffPrefix]:
    """Compares the column names of two DataFrames.

    Returns a list of column names that preserves the ordering of the left DataFrame when possible.
    The columns names are prefixed by a character according to the following convention:

    - ' ' if the column exists in both DataFrame
    - '-' if it only exists in the left DataFrame
    - '+' if it only exists in the right DataFrame

    Args:
        left_col_names: A list
        right_col_names: Another DataFrame

    Returns:
        A list of column names prefixed with a character: ' ', '+' or '-'

    Examples:

        >>> left_cols = ["id", "col1", "col2", "col3"]
        >>> right_cols = ["id", "col1", "col4", "col3"]
        >>> _diff_dataframe_column_names(left_cols, right_cols)
        {'id': ' ', 'col1': ' ', 'col2': '-', 'col4': '+', 'col3': ' '}
        >>> _diff_dataframe_column_names(left_cols, left_cols)
        {'id': ' ', 'col1': ' ', 'col2': ' ', 'col3': ' '}

        >>> left_cols = ["id", "col1", "col2", "col3", "col4"]
        >>> right_cols = ["id", "col1", "col4", "col3", "col2"]
        >>> _diff_dataframe_column_names(left_cols, right_cols)
        {'id': ' ', 'col1': ' ', 'col2': ' ', 'col3': ' ', 'col4': ' '}

    """
    diff = list(difflib.unified_diff(left_col_names, right_col_names, n=10000))[2:]
    same_schema = len(diff) == 0
    if same_schema:
        list_result = [DiffPrefix.UNCHANGED + s for s in left_col_names]
    else:
        list_result = _remove_potential_duplicates_from_diff(diff[1:])
    return {s[1:]: DiffPrefix(s[0]) for s in list_result}
