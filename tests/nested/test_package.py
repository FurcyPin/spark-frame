from typing import Callable, Dict

from pyspark.sql import Column, SparkSession
from pyspark.sql import functions as f

from spark_frame.fp.printable_function import PrintableFunction
from spark_frame.nested_impl.package import (
    _build_nested_struct_tree,
    _build_transformation_from_tree,
    resolve_nested_fields,
)
from spark_frame.utils import schema_string, show_string, strip_margin


def replace_named_functions_with_functions(
    transformations: Dict[str, PrintableFunction]
) -> Dict[str, Callable[[Column], Column]]:
    return {alias: transformation.func for alias, transformation in transformations.items()}


class TestBuildTransformationFromTree:
    """The purpose of this test class is mostly to make sure that PrintableFunctions are properly printed
    The code's logic is mostly tested in TestResolveNestedColumns.
    """

    def test_value_with_string_expr_and_string_alias(self, spark: SparkSession):
        """
        GIVEN a transformation that returns a string expression and has an alias of type str
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {"a": PrintableFunction(lambda s: "a", "a")}
        transformations = {"a": "a"}
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        actual = _build_transformation_from_tree(_build_nested_struct_tree(transformations))
        assert str(named_actual) == """lambda x: [a.alias(a)]"""
        assert str(actual) == """lambda x: [f.col('a').alias(a)]"""

    def test_value_with_col_expr(self, spark: SparkSession):
        """
        GIVEN a transformation that returns a column
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {"a": PrintableFunction(lambda s: f.col("a"), lambda s: 'f.col("a")')}
        transformations = {"a": f.col("a")}
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        actual = _build_transformation_from_tree(_build_nested_struct_tree(transformations))
        assert str(named_actual) == """lambda x: [f.col("a").alias(a)]"""
        assert str(actual) == """lambda x: [Column<'a'>.alias(a)]"""

    def test_value_with_aliased_col_expr(self, spark: SparkSession):
        """
        GIVEN a transformation that returns an aliased column
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "a": PrintableFunction(
                lambda s: f.col("a").alias("other_alias"), lambda s: 'f.col("a").alias("other_alias")'
            )
        }
        transformations = {"a": f.col("a").alias("other_alias")}
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        actual = _build_transformation_from_tree(_build_nested_struct_tree(transformations))
        assert str(named_actual) == """lambda x: [f.col("a").alias("other_alias").alias(a)]"""
        assert str(actual) == """lambda x: [Column<'a AS other_alias'>.alias(a)]"""

    def test_struct(self, spark: SparkSession):
        """
        GIVEN a transformation on a struct
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f"""{s}["a"].cast("DOUBLE")""")
        }
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        assert str(named_actual) == """lambda x: [f.struct([x['s']["a"].cast("DOUBLE").alias(a)]).alias(s)]"""

    def test_struct_with_static_expression(self, spark: SparkSession):
        """
        GIVEN a transformation on a struct that uses the full name of one of the struct's field (s.a)
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s.a": PrintableFunction(lambda s: f.col("s.a").cast("DOUBLE"), lambda s: """f.col("s.a").cast("DOUBLE")""")
        }
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        assert str(named_actual) == """lambda x: [f.struct([f.col("s.a").cast("DOUBLE").alias(a)]).alias(s)]"""

    def test_array(self, spark: SparkSession):
        """
        GIVEN a transformation on an array
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {"e!": PrintableFunction(lambda e: e.cast("DOUBLE"), lambda e: f'{e}.cast("DOUBLE")')}
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        assert str(named_actual) == """lambda x: [f.transform(x['e'], lambda x: x.cast("DOUBLE")).alias(e)]"""

    def test_array_struct(self, spark: SparkSession):
        """
        GIVEN a transformation on an array<struct>
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s!.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")')
        }
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        assert str(named_actual) == (
            """lambda x: [f.transform(x['s'], lambda x: f.struct([x["a"].cast("DOUBLE").alias(a)])).alias(s)]"""
        )

    def test_struct_in_struct(self, spark: SparkSession):
        """
        GIVEN a transformation on a struct inside a struct
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s1.s2.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")')
        }
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        assert str(named_actual) == (
            """lambda x: [f.struct([f.struct([x['s1']['s2']["a"].cast("DOUBLE").alias(a)]).alias(s2)]).alias(s1)]"""
        )

    def test_array_in_struct(self, spark: SparkSession):
        """
        GIVEN a transformation on an array inside a struct
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s1.e!": PrintableFunction(lambda e: e.cast("DOUBLE"), lambda e: f'{e}.cast("DOUBLE")'),
        }
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        assert str(named_actual) == (
            "lambda x: [f.struct([f.transform(x['s1']['e'], lambda x: " 'x.cast("DOUBLE")).alias(e)]).alias(s1)]'
        )

    def test_array_struct_in_struct(self, spark: SparkSession):
        """
        GIVEN a transformation on an array<struct> inside a struct
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s1.s2!.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")'),
        }
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        assert str(named_actual) == (
            "lambda x: [f.struct([f.transform(x['s1']['s2'], lambda x: "
            """f.struct([x["a"].cast("DOUBLE").alias(a)])).alias(s2)]).alias(s1)]"""
        )

    def test_array_in_array(self, spark: SparkSession):
        """
        GIVEN a transformation on an array inside an array
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {"e!!": PrintableFunction(lambda e: e.cast("DOUBLE"), lambda e: f'{e}.cast("DOUBLE")')}
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        assert str(named_actual) == (
            "lambda x: [f.transform(x['e'], lambda x: f.transform(x, lambda x: " 'x.cast("DOUBLE"))).alias(e)]'
        )

    def test_array_struct_in_array(self, spark: SparkSession):
        """
        GIVEN a transformation on array<struct> inside an array
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s!!.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")'),
        }
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        assert str(named_actual) == (
            "lambda x: [f.transform(x['s'], lambda x: f.transform(x, lambda x: "
            'f.struct([x["a"].cast("DOUBLE").alias(a)]))).alias(s)]'
        )

    def test_struct_in_array_struct(self, spark: SparkSession):
        """
        GIVEN a transformation on struct inside an array<struct>
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s1!.s2.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")'),
        }
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        assert str(named_actual) == (
            "lambda x: [f.transform(x['s1'], lambda x: "
            """f.struct([f.struct([x['s2']["a"].cast("DOUBLE").alias(a)]).alias(s2)])).alias(s1)]"""
        )

    def test_array_in_array_struct(self, spark: SparkSession):
        """
        GIVEN a transformation on an array inside an array<struct>
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s1!.e!": PrintableFunction(lambda e: e.cast("DOUBLE"), lambda e: f'{e}.cast("DOUBLE")')
        }
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        assert str(named_actual) == (
            "lambda x: [f.transform(x['s1'], lambda x: f.struct([f.transform(x['e'], lambda "
            'x: x.cast("DOUBLE")).alias(e)])).alias(s1)]'
        )

    def test_array_struct_in_array_struct(self, spark: SparkSession):
        """
        GIVEN a transformation on an array<struct> inside an array<struct>
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s1!.s2!.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")')
        }
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations))
        assert str(named_actual) == (
            "lambda x: [f.transform(x['s1'], lambda x: f.struct([f.transform(x['s2'], lambda "
            'x: f.struct([x["a"].cast("DOUBLE").alias(a)])).alias(s2)])).alias(s1)]'
        )

    def test_array_struct_in_array_struct_with_sort(self, spark: SparkSession):
        """
        GIVEN a transformation on an array<struct> inside an array<struct>
        WHEN we print the PrintableFunction generated in resolve_nested_columns using sort = True
        THEN the result should be human-readable
        """
        named_transformations = {
            "s1!.s2!.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")')
        }
        named_actual = _build_transformation_from_tree(_build_nested_struct_tree(named_transformations), sort=True)
        assert str(named_actual) == (
            "lambda x: [f.sort_array(f.transform(x['s1'], lambda x: "
            "f.struct([f.sort_array(f.transform(x['s2'], lambda x: "
            'f.struct([x["a"].cast("DOUBLE").alias(a)]))).alias(s2)]))).alias(s1)]'
        )


class TestResolveNestedFields:
    def test_value_with_string_expr(self, spark: SparkSession):
        """
        GIVEN a DataFrame with a simple value
        WHEN we use resolve_nested_columns on it with a string expression
        THEN the transformation should work
        """
        df = spark.sql("SELECT 1 as a")
        assert show_string(df) == strip_margin(
            """
            |+---+
            ||  a|
            |+---+
            ||  1|
            |+---+
            |"""
        )
        named_transformations = {"a": PrintableFunction(lambda s: "a", lambda s: '"a"')}
        transformations = {"a": "a"}
        expected = strip_margin(
            """
            |+---+
            ||  a|
            |+---+
            ||  1|
            |+---+
            |"""
        )
        assert show_string(df.select(*resolve_nested_fields(named_transformations))) == expected
        assert show_string(df.select(*resolve_nested_fields(transformations))) == expected

    def test_value_with_col_expr(self, spark: SparkSession):
        """
        GIVEN a DataFrame with a simple value
        WHEN we use resolve_nested_columns on it with a column expression
        THEN the transformation should work
        """
        df = spark.sql("SELECT 1 as a")
        assert show_string(df) == strip_margin(
            """
            |+---+
            ||  a|
            |+---+
            ||  1|
            |+---+
            |"""
        )
        named_transformations = {"a": PrintableFunction(lambda s: f.col("a"), lambda s: 'f.col("a")')}
        transformations = {"a": f.col("a")}
        expected = strip_margin(
            """
            |+---+
            ||  a|
            |+---+
            ||  1|
            |+---+
            |"""
        )
        assert show_string(df.select(*resolve_nested_fields(named_transformations))) == expected
        assert show_string(df.select(*resolve_nested_fields(transformations))) == expected

    def test_value_with_aliased_col_expr(self, spark: SparkSession):
        """
        GIVEN a DataFrame with a simple value
        WHEN we use resolve_nested_columns on it with a column expression using a different alias
        THEN the transformation should work and the alias should be ignored
        """
        df = spark.sql("SELECT 1 as a")
        assert show_string(df) == strip_margin(
            """
            |+---+
            ||  a|
            |+---+
            ||  1|
            |+---+
            |"""
        )
        named_transformations = {
            "a": PrintableFunction(
                lambda s: f.col("a").alias("other_alias"), lambda s: 'f.col("a").alias("other_alias")'
            )
        }
        transformations = {"a": f.col("a").alias("other_alias")}
        expected = strip_margin(
            """
            |+---+
            ||  a|
            |+---+
            ||  1|
            |+---+
            |"""
        )
        assert show_string(df.select(*resolve_nested_fields(named_transformations))) == expected
        assert show_string(df.select(*resolve_nested_fields(transformations))) == expected

    def test_struct(self, spark: SparkSession):
        """
        GIVEN a DataFrame with a struct
        WHEN we use resolve_nested_columns on it
        THEN the transformation should work
        """
        df = spark.sql("SELECT STRUCT(2 as a) as s")
        assert show_string(df) == strip_margin(
            """
            |+---+
            ||  s|
            |+---+
            ||{2}|
            |+---+
            |"""
        )
        named_transformations = {
            "s.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f"""{s}["a"].cast("DOUBLE")""")
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected = strip_margin(
            """
            |+-----+
            ||    s|
            |+-----+
            ||{2.0}|
            |+-----+
            |"""
        )
        assert show_string(df.select(*resolve_nested_fields(named_transformations))) == expected
        assert show_string(df.select(*resolve_nested_fields(transformations))) == expected

    def test_struct_with_static_expression(self, spark: SparkSession):
        """
        GIVEN a DataFrame with a struct
        WHEN we use resolve_nested_columns on it without lambda expression
        THEN the transformation should work
        """
        df = spark.sql("SELECT STRUCT(2 as a) as s")
        assert show_string(df) == strip_margin(
            """
            |+---+
            ||  s|
            |+---+
            ||{2}|
            |+---+
            |"""
        )
        named_transformations = {
            "s.a": PrintableFunction(lambda s: f.col("s.a").cast("DOUBLE"), lambda s: """f.col("s.a").cast("DOUBLE")""")
        }
        transformations = {"s.a": f.col("s.a").cast("DOUBLE")}
        expected = strip_margin(
            """
            |+-----+
            ||    s|
            |+-----+
            ||{2.0}|
            |+-----+
            |"""
        )
        assert show_string(df.select(*resolve_nested_fields(named_transformations))) == expected
        assert show_string(df.select(*resolve_nested_fields(transformations))) == expected

    def test_array(self, spark: SparkSession):
        """
        GIVEN a DataFrame with an array
        WHEN we use resolve_nested_columns on it
        THEN the transformation should work
        """
        df = spark.sql("SELECT ARRAY(2, 3) as e")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- e: array (nullable = false)
            | |    |-- element: integer (containsNull = false)
            |"""
        )
        assert show_string(df) == strip_margin(
            """
            |+------+
            ||     e|
            |+------+
            ||[2, 3]|
            |+------+
            |"""
        )
        named_transformations = {"e!": PrintableFunction(lambda e: e.cast("DOUBLE"), lambda e: f'{e}.cast("DOUBLE")')}
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- e: array (nullable = false)
            | |    |-- element: double (containsNull = false)
            |"""
        )
        expected = strip_margin(
            """
            |+----------+
            ||         e|
            |+----------+
            ||[2.0, 3.0]|
            |+----------+
            |"""
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_array_struct(self, spark: SparkSession):
        """
        GIVEN a DataFrame with an array<struct>
        WHEN we use resolve_nested_columns on it
        THEN the transformation should work
        """
        df = spark.sql("SELECT ARRAY(STRUCT(2 as a)) as s")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- s: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- a: integer (nullable = false)
            |"""
        )
        assert show_string(df) == strip_margin(
            """
            |+-----+
            ||    s|
            |+-----+
            ||[{2}]|
            |+-----+
            |"""
        )
        named_transformations = {
            "s!.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")')
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- a: double (nullable = false)
            |"""
        )
        expected = strip_margin(
            """
            |+-------+
            ||      s|
            |+-------+
            ||[{2.0}]|
            |+-------+
            |"""
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_struct_in_struct(self, spark: SparkSession):
        """
        GIVEN a DataFrame with a struct inside a struct
        WHEN we use resolve_nested_columns on it
        THEN the transformation should work
        """
        df = spark.sql("SELECT STRUCT(STRUCT(2 as a) as s2) as s1")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- s1: struct (nullable = false)
            | |    |-- s2: struct (nullable = false)
            | |    |    |-- a: integer (nullable = false)
            |"""
        )
        assert show_string(df) == strip_margin(
            """
            |+-----+
            ||   s1|
            |+-----+
            ||{{2}}|
            |+-----+
            |"""
        )
        named_transformations = {
            "s1.s2.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")')
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1: struct (nullable = false)
            | |    |-- s2: struct (nullable = false)
            | |    |    |-- a: double (nullable = false)
            |"""
        )
        expected = strip_margin(
            """
            |+-------+
            ||     s1|
            |+-------+
            ||{{2.0}}|
            |+-------+
            |"""
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_array_in_struct(self, spark: SparkSession):
        """
        GIVEN a DataFrame with an array inside a struct
        WHEN we use resolve_nested_columns on it
        THEN the transformation should work
        """
        df = spark.sql("SELECT STRUCT(ARRAY(2, 3) as e) as s1")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- s1: struct (nullable = false)
            | |    |-- e: array (nullable = false)
            | |    |    |-- element: integer (containsNull = false)
            |"""
        )
        assert show_string(df) == strip_margin(
            """
            |+--------+
            ||      s1|
            |+--------+
            ||{[2, 3]}|
            |+--------+
            |"""
        )
        named_transformations = {
            "s1.e!": PrintableFunction(lambda e: e.cast("DOUBLE"), lambda e: f'{e}.cast("DOUBLE")'),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1: struct (nullable = false)
            | |    |-- e: array (nullable = false)
            | |    |    |-- element: double (containsNull = false)
            |"""
        )
        expected = strip_margin(
            """
            |+------------+
            ||          s1|
            |+------------+
            ||{[2.0, 3.0]}|
            |+------------+
            |"""
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_array_struct_in_struct(self, spark: SparkSession):
        """
        GIVEN a DataFrame with an array<struct> inside a struct
        WHEN we use resolve_nested_columns on it
        THEN the transformation should work
        """
        df = spark.sql("SELECT STRUCT(ARRAY(STRUCT(2 as a)) as s2) as s1")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- s1: struct (nullable = false)
            | |    |-- s2: array (nullable = false)
            | |    |    |-- element: struct (containsNull = false)
            | |    |    |    |-- a: integer (nullable = false)
            |"""
        )
        assert show_string(df) == strip_margin(
            """
            |+-------+
            ||     s1|
            |+-------+
            ||{[{2}]}|
            |+-------+
            |"""
        )
        named_transformations = {
            "s1.s2!.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")'),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1: struct (nullable = false)
            | |    |-- s2: array (nullable = false)
            | |    |    |-- element: struct (containsNull = false)
            | |    |    |    |-- a: double (nullable = false)
            |"""
        )
        expected = strip_margin(
            """
            |+---------+
            ||       s1|
            |+---------+
            ||{[{2.0}]}|
            |+---------+
            |"""
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_array_in_array(self, spark: SparkSession):
        """
        GIVEN a DataFrame with an array inside an array
        WHEN we use resolve_nested_columns on it
        THEN the transformation should work
        """
        df = spark.sql("SELECT ARRAY(ARRAY(1)) as e")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- e: array (nullable = false)
            | |    |-- element: array (containsNull = false)
            | |    |    |-- element: integer (containsNull = false)
            |"""
        )
        assert show_string(df) == strip_margin(
            """
            |+-----+
            ||    e|
            |+-----+
            ||[[1]]|
            |+-----+
            |"""
        )
        named_transformations = {"e!!": PrintableFunction(lambda e: e.cast("DOUBLE"), lambda e: f'{e}.cast("DOUBLE")')}
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- e: array (nullable = false)
            | |    |-- element: array (containsNull = false)
            | |    |    |-- element: double (containsNull = false)
            |"""
        )
        expected = strip_margin(
            """
            |+-------+
            ||      e|
            |+-------+
            ||[[1.0]]|
            |+-------+
            |"""
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_array_struct_in_array(self, spark: SparkSession):
        """
        GIVEN a DataFrame with an array<struct> inside an array
        WHEN we use resolve_nested_columns on it
        THEN the transformation should work
        """
        df = spark.sql("SELECT ARRAY(ARRAY(STRUCT(1 as a))) as s")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- s: array (nullable = false)
            | |    |-- element: array (containsNull = false)
            | |    |    |-- element: struct (containsNull = false)
            | |    |    |    |-- a: integer (nullable = false)
            |"""
        )
        assert show_string(df) == strip_margin(
            """
            |+-------+
            ||      s|
            |+-------+
            ||[[{1}]]|
            |+-------+
            |"""
        )
        named_transformations = {
            "s!!.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")'),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s: array (nullable = false)
            | |    |-- element: array (containsNull = false)
            | |    |    |-- element: struct (containsNull = false)
            | |    |    |    |-- a: double (nullable = false)
            |"""
        )
        expected = strip_margin(
            """
            |+---------+
            ||        s|
            |+---------+
            ||[[{1.0}]]|
            |+---------+
            |"""
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_struct_in_array_struct(self, spark: SparkSession):
        """
        GIVEN a DataFrame with a struct inside an array<struct>
        WHEN we use resolve_nested_columns on it
        THEN the transformation should work
        """
        df = spark.sql("SELECT ARRAY(STRUCT(STRUCT(2 as a) as s2)) as s1")
        assert schema_string(df) == strip_margin(
            """
        |root
        | |-- s1: array (nullable = false)
        | |    |-- element: struct (containsNull = false)
        | |    |    |-- s2: struct (nullable = false)
        | |    |    |    |-- a: integer (nullable = false)
        |"""
        )
        assert show_string(df) == strip_margin(
            """
        |+-------+
        ||     s1|
        |+-------+
        ||[{{2}}]|
        |+-------+
        |"""
        )
        named_transformations = {
            "s1!.s2.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")'),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- s2: struct (nullable = false)
            | |    |    |    |-- a: double (nullable = false)
            |"""
        )
        expected = strip_margin(
            """
            |+---------+
            ||       s1|
            |+---------+
            ||[{{2.0}}]|
            |+---------+
            |"""
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_array_in_array_struct(self, spark: SparkSession):
        """
        GIVEN a DataFrame with an array inside an array<struct>
        WHEN we use resolve_nested_columns on it
        THEN the transformation should work
        """
        df = spark.sql("SELECT ARRAY(STRUCT(ARRAY(2, 3) as e)) as s1")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- s1: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- e: array (nullable = false)
            | |    |    |    |-- element: integer (containsNull = false)
            |"""
        )
        assert show_string(df) == strip_margin(
            """
            |+----------+
            ||        s1|
            |+----------+
            ||[{[2, 3]}]|
            |+----------+
            |"""
        )
        named_transformations = {
            "s1!.e!": PrintableFunction(lambda e: e.cast("DOUBLE"), lambda e: f'{e}.cast("DOUBLE")')
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- e: array (nullable = false)
            | |    |    |    |-- element: double (containsNull = false)
            |"""
        )
        expected = strip_margin(
            """
            |+--------------+
            ||            s1|
            |+--------------+
            ||[{[2.0, 3.0]}]|
            |+--------------+
            |"""
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_array_struct_in_array_struct(self, spark: SparkSession):
        """
        GIVEN a DataFrame with an array<struct> inside another array<struct>
        WHEN we use resolve_nested_columns on it
        THEN the transformation should work
        """
        df = spark.sql("SELECT ARRAY(STRUCT(ARRAY(STRUCT(2 as a)) as s2)) as s1")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- s1: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- s2: array (nullable = false)
            | |    |    |    |-- element: struct (containsNull = false)
            | |    |    |    |    |-- a: integer (nullable = false)
            |"""
        )
        assert show_string(df) == strip_margin(
            """
            |+---------+
            ||       s1|
            |+---------+
            ||[{[{2}]}]|
            |+---------+
            |"""
        )
        named_transformations = {
            "s1!.s2!.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")')
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- s2: array (nullable = false)
            | |    |    |    |-- element: struct (containsNull = false)
            | |    |    |    |    |-- a: double (nullable = false)
            |"""
        )
        expected = strip_margin(
            """
            |+-----------+
            ||         s1|
            |+-----------+
            ||[{[{2.0}]}]|
            |+-----------+
            |"""
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_array_struct_in_array_struct_with_sort(self, spark: SparkSession):
        """
        GIVEN a DataFrame with an array<struct> inside another array<struct>
        WHEN we use resolve_nested_columns on it with the sort option activated
        THEN the transformation should work
        """
        df = spark.sql(
            """
                SELECT ARRAY(
                    STRUCT(ARRAY(STRUCT(4 as a), STRUCT(3 as a)) as s2),
                    STRUCT(ARRAY(STRUCT(5 as a), STRUCT(2 as a)) as s2)
                ) as s1
            """
        )
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- s1: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- s2: array (nullable = false)
            | |    |    |    |-- element: struct (containsNull = false)
            | |    |    |    |    |-- a: integer (nullable = false)
            |"""
        )
        assert show_string(df, truncate=False) == strip_margin(
            """
            |+----------------------------+
            ||s1                          |
            |+----------------------------+
            ||[{[{4}, {3}]}, {[{5}, {2}]}]|
            |+----------------------------+
            |"""
        )
        named_transformations = {
            "s1!.s2!.a": PrintableFunction(lambda s: s["a"].cast("DOUBLE"), lambda s: f'{s}["a"].cast("DOUBLE")')
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- s2: array (nullable = false)
            | |    |    |    |-- element: struct (containsNull = false)
            | |    |    |    |    |-- a: double (nullable = false)
            |"""
        )
        expected = strip_margin(
            """
            |+------------------------------------+
            ||s1                                  |
            |+------------------------------------+
            ||[{[{2.0}, {5.0}]}, {[{3.0}, {4.0}]}]|
            |+------------------------------------+
            |"""
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations, sort=True))
        actual = df.select(*resolve_nested_fields(transformations, sort=True))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named, truncate=False) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual, truncate=False) == expected
