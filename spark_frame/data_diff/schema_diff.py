import difflib
from dataclasses import dataclass
from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from spark_frame.data_type_utils import flatten_schema, is_nullable


@dataclass
class SchemaDiffResult:
    same_schema: bool
    diff_str: str
    nb_cols: int
    left_schema_str: str
    right_schema_str: str

    def display(self) -> None:
        if not self.same_schema:
            print(f"Schema has changed:\n{self.diff_str}")
            print("WARNING: columns that do not match both sides will be ignored")
        else:
            print(f"Schema: ok ({self.nb_cols})")


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
    """
    res = []
    for field in schema:
        s = f"{field.name} {field.dataType.simpleString().upper()}"
        if include_nullable:
            if is_nullable(field):
                s += " (nullable)"
            else:
                s += " (not nullable)"
        if include_metadata:
            s += f" {field.metadata}"
        res.append(s)
    return res


def diff_dataframe_schemas(left_df: DataFrame, right_df: DataFrame) -> SchemaDiffResult:
    """Compares two DataFrames schemas and print out the differences.
    Ignore the nullable and comment attributes.

    Args:
        left_df: A Spark DataFrame
        right_df: Another DataFrame

    Returns:
        A SchemaDiffResult object

    Examples:

        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> left_df = spark.sql('''SELECT 1 as id, "" as c1, "" as c2, ARRAY(STRUCT(2 as a, "" as b)) as c4''')
        >>> right_df = spark.sql('''SELECT 1 as id, 2 as c1, "" as c3, ARRAY(STRUCT(3 as a, "" as d)) as c4''')
        >>> schema_diff_result = diff_dataframe_schemas(left_df, right_df)
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

    """
    left_schema_flat = flatten_schema(left_df.schema, explode=True)
    right_schema_flat = flatten_schema(right_df.schema, explode=True)
    left_schema: List[str] = _schema_to_string(left_schema_flat)
    right_schema: List[str] = _schema_to_string(right_schema_flat)

    diff_str = list(difflib.unified_diff(left_schema, right_schema, n=10000))[2:]
    same_schema = len(diff_str) == 0
    if same_schema:
        diff_str = left_schema
    return SchemaDiffResult(
        same_schema=same_schema,
        diff_str="\n".join(diff_str),
        nb_cols=len(left_df.columns),
        left_schema_str="\n".join(left_schema),
        right_schema_str="\n".join(right_schema),
    )
