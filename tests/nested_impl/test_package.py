from typing import Callable, Dict

import pytest
from pyspark.sql import Column, SparkSession
from pyspark.sql import functions as f

import spark_frame.exceptions
import spark_frame.utils
from spark_frame import nested
from spark_frame.fp.higher_order import identity
from spark_frame.fp.printable_function import PrintableFunction
from spark_frame.nested_impl.package import (
    _build_nested_struct_tree,
    _build_transformation_from_tree,
    resolve_nested_fields,
)
from spark_frame.utils import schema_string, show_string, strip_margin


def replace_named_functions_with_functions(
    transformations: Dict[str, PrintableFunction],
) -> Dict[str, Callable[[Column], Column]]:
    return {alias: transformation.func for alias, transformation in transformations.items()}


class TestBuildTransformationFromTree:
    """The purpose of this test class is mostly to make sure that PrintableFunctions are properly printed
    The code's logic is mostly tested in TestResolveNestedColumns.
    """

    def test_value_with_string_expr_and_string_alias(self):
        """
        GIVEN a transformation that returns a string expression and has an alias of type str
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {"a": PrintableFunction(lambda s: "a", "a")}
        transformations = {"a": "a"}
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        actual = _build_transformation_from_tree(
            _build_nested_struct_tree(transformations),
        )
        assert str(actual_named) == """lambda x: [a.alias('a')]"""
        assert str(actual) == """lambda x: [f.col('a').alias('a')]"""

    def test_value_with_col_expr(self):
        """
        GIVEN a transformation that returns a column
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "a": PrintableFunction(lambda s: f.col("a"), lambda s: 'f.col("a")'),
        }
        transformations = {"a": f.col("a")}
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        actual = _build_transformation_from_tree(
            _build_nested_struct_tree(transformations),
        )
        assert str(actual_named) == """lambda x: [f.col("a").alias('a')]"""
        assert str(actual) == """lambda x: [Column<'a'>.alias('a')]"""

    def test_value_with_aliased_col_expr(self):
        """
        GIVEN a transformation that returns an aliased column
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "a": PrintableFunction(
                lambda s: f.col("a").alias("other_alias"),
                lambda s: 'f.col("a").alias("other_alias")',
            ),
        }
        transformations = {"a": f.col("a").alias("other_alias")}
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        actual = _build_transformation_from_tree(
            _build_nested_struct_tree(transformations),
        )
        assert str(actual_named) == """lambda x: [f.col("a").alias("other_alias").alias('a')]"""
        assert str(actual) == """lambda x: [Column<'a AS other_alias'>.alias('a')]"""

    def test_struct(self):
        """
        GIVEN a transformation on a struct
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s.a": PrintableFunction(
                lambda s: f.col("s.a").cast("DOUBLE"),
                lambda s: """f.col("s.a").cast("DOUBLE")""",
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == """lambda x: [f.struct([f.col("s.a").cast("DOUBLE").alias('a')]).alias('s')]"""

    def test_struct_with_static_expression(self):
        """
        GIVEN a transformation on a struct that uses the full name of one of the struct's field (s.a)
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s.a": PrintableFunction(
                lambda s: f.col("s.a").cast("DOUBLE"),
                lambda s: """f.col("s.a").cast("DOUBLE")""",
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == """lambda x: [f.struct([f.col("s.a").cast("DOUBLE").alias('a')]).alias('s')]"""

    def test_array(self):
        """
        GIVEN a transformation on an array
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "e!": PrintableFunction(
                lambda e: e.cast("DOUBLE"),
                lambda e: f'{e}.cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == """lambda x: [f.transform(x['e'], lambda x: x.cast("DOUBLE")).alias('e')]"""

    def test_array_struct(self):
        """
        GIVEN a transformation on an array<struct>
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s!.a": PrintableFunction(
                lambda s: s["a"].cast("DOUBLE"),
                lambda s: f'{s}["a"].cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            """lambda x: [f.transform(x['s'], lambda x: f.struct([x["a"].cast("DOUBLE").alias('a')])).alias('s')]"""
        )

    def test_struct_in_struct(self):
        """
        GIVEN a transformation on a struct inside a struct
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s1.s2.a": PrintableFunction(
                lambda s: f.col("s1")["s2"]["a"].cast("DOUBLE"),
                lambda s: 'f.col("s1")["s2"]["a"].cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            """lambda x: [f.struct([f.struct([f.col("s1")["s2"]["a"]"""
            """.cast("DOUBLE").alias('a')]).alias('s2')]).alias('s1')]"""
        )

    def test_array_in_struct(self):
        """
        GIVEN a transformation on an array inside a struct
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s1.e!": PrintableFunction(
                lambda e: e.cast("DOUBLE"),
                lambda e: f'{e}.cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            """lambda x: [f.struct([f.transform(x['s1']['e'], lambda x: x.cast("DOUBLE")).alias('e')]).alias('s1')]"""
        )

    def test_array_struct_in_struct(self):
        """
        GIVEN a transformation on an array<struct> inside a struct
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s1.s2!.a": PrintableFunction(
                lambda s: s["a"].cast("DOUBLE"),
                lambda s: f'{s}["a"].cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            "lambda x: [f.struct([f.transform(x['s1']['s2'], lambda x: "
            """f.struct([x["a"].cast("DOUBLE").alias('a')])).alias('s2')]).alias('s1')]"""
        )

    def test_array_in_array(self):
        """
        GIVEN a transformation on an array inside an array
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "e!!": PrintableFunction(
                lambda e: e.cast("DOUBLE"),
                lambda e: f'{e}.cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            """lambda x: [f.transform(x['e'], lambda x: f.transform(x, lambda x: x.cast("DOUBLE"))).alias('e')]"""
        )

    def test_array_struct_in_array(self):
        """
        GIVEN a transformation on array<struct> inside an array
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s!!.a": PrintableFunction(
                lambda s: s["a"].cast("DOUBLE"),
                lambda s: f'{s}["a"].cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            "lambda x: [f.transform(x['s'], lambda x: f.transform(x, lambda x: "
            """f.struct([x["a"].cast("DOUBLE").alias('a')]))).alias('s')]"""
        )

    def test_struct_in_array_struct(self):
        """
        GIVEN a transformation on struct inside an array<struct>
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s1!.s2.a": PrintableFunction(
                lambda s: s["a"].cast("DOUBLE"),
                lambda s: f'{s}["a"].cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            "lambda x: [f.transform(x['s1'], lambda x: "
            """f.struct([f.struct([x["a"].cast("DOUBLE").alias('a')]).alias('s2')])).alias('s1')]"""
        )

    def test_array_in_array_struct(self):
        """
        GIVEN a transformation on an array inside an array<struct>
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s1!.e!": PrintableFunction(
                lambda e: e.cast("DOUBLE"),
                lambda e: f'{e}.cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            "lambda x: [f.transform(x['s1'], lambda x: f.struct([f.transform(x['e'], lambda "
            """x: x.cast("DOUBLE")).alias('e')])).alias('s1')]"""
        )

    def test_array_struct_in_array_struct(self):
        """
        GIVEN a transformation on an array<struct> inside an array<struct>
        WHEN we print the PrintableFunction generated in resolve_nested_columns
        THEN the result should be human-readable
        """
        named_transformations = {
            "s1!.s2!.a": PrintableFunction(
                lambda s: s["a"].cast("DOUBLE"),
                lambda s: f'{s}["a"].cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            "lambda x: [f.transform(x['s1'], lambda x: f.struct([f.transform(x['s2'], lambda "
            """x: f.struct([x["a"].cast("DOUBLE").alias('a')])).alias('s2')])).alias('s1')]"""
        )

    def test_array_struct_in_array_struct_with_transformation_using_field_from_first_array(
        self,
    ):
        """
        GIVEN a DataFrame with an array<struct> inside another array<struct>
        WHEN we use resolve_nested_columns on it with a transformation that uses data from the first array
        THEN the transformation should work
        """
        named_transformations = {
            "s1!.a": PrintableFunction(lambda s1: s1["a"], lambda s1: f'{s1}["a"]'),
            "s1!.s2!.b": PrintableFunction(
                lambda s1, s2: (s1["a"] + s2["b"]).cast("DOUBLE"),
                lambda s1, s2: f'({s1}["a"]+{s2}["b"]).cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            """lambda x: [f.transform(x['s1'], lambda x: f.struct([x["a"].alias('a'), f.transform(x['s2'], """
            """lambda x: f.struct([(x["a"]+x["b"]).cast("DOUBLE").alias('b')])).alias('s2')])).alias('s1')]"""
        )

    def test_struct_in_array_struct_in_array_struct(self):
        """
        GIVEN a DataFrame with a struct inside an array<struct> inside an array<struct>
        WHEN we use resolve_nested_columns with a transformation that access an element from the outermost struct
        THEN the transformation should work
        """
        # Here, the lambda function is applied to the elements of `s2`, not `s3`
        named_transformations = {
            "s1!.s2!.a": PrintableFunction(lambda s2: s2["a"], lambda s2: f'{s2}["a"]'),
            "s1!.s2!.s3.b": PrintableFunction(
                lambda s2: s2["a"].cast("DOUBLE"),
                lambda s2: f'{s2}["a"].cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            "lambda x: [f.transform(x['s1'], lambda x: f.struct([f.transform(x['s2'], "
            """lambda x: f.struct([x["a"].alias('a'), """
            """f.struct([x["a"].cast("DOUBLE").alias('b')]).alias('s3')])).alias('s2')])).alias('s1')]"""
        )

    def test_struct_in_struct_in_array_struct_in_struct_in_array_struct(self):
        """
        GIVEN a DataFrame with a struct inside a struct inside an array<struct> inside a struct inside an array<struct>
        WHEN we use resolve_nested_columns with a transformation that access an element from the outermost struct
        THEN the transformation should work
        """
        # Here, the lambda function is applied to the elements of `s3`, not `s4` or `s5`
        named_transformations = {
            "s1!.s2.s3!.s4.a": PrintableFunction(
                lambda s3: s3["s4"]["a"],
                lambda s3: f'{s3}["s4"]["a"]',
            ),
            "s1!.s2.s3!.s4.s5.b": PrintableFunction(
                lambda s3: s3["s4"]["a"].cast("DOUBLE"),
                lambda s3: f'{s3}["s4"]["a"].cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            "lambda x: [f.transform(x['s1'], lambda x: "
            "f.struct([f.struct([f.transform(x['s2']['s3'], lambda x: "
            """f.struct([f.struct([x["s4"]["a"].alias('a'), """
            """f.struct([x["s4"]["a"].cast("DOUBLE").alias('b')]).alias('s5')])"""
            """.alias('s4')])).alias('s3')]).alias('s2')])).alias('s1')]"""
        )

    def test_struct_in_map_values(self):
        """
        GIVEN a DataFrame with a struct inside map values
        WHEN we use resolve_nested_columns on it
        THEN the transformation should work
        """
        named_transformations = {
            "m1%key": identity,
            "m1%value.a": PrintableFunction(
                lambda value: value["a"].cast("DOUBLE"),
                lambda s: f'{s}["a"].cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            "lambda x: [f.transform_values(f.transform_keys(x['m1'], lambda k, v: k)),lambda k, v: "
            """f.struct([v["a"].cast("DOUBLE").alias('a')])).alias('m1')]"""
        )

    def test_struct_in_map_values_in_array_struct_with_transformation_using_field_from_first_array(
        self,
    ):
        """
        GIVEN a DataFrame with a struct inside map values inside an array<struct>
        WHEN we use resolve_nested_columns on it with a transformation that uses data from the first array
        THEN the transformation should work
        """
        named_transformations = {
            "s1!.a": None,
            "s1!.m1%key.b": PrintableFunction(
                lambda s1, key: (s1["a"] + key["b"]).cast("DOUBLE"),
                lambda s1, key: f'({s1}["a"] + {key}["b"]).cast("DOUBLE")',
            ),
            "s1!.m1%value.c": PrintableFunction(
                lambda s1, value: (s1["a"] + value["c"]).cast("DOUBLE"),
                lambda s1, value: f'({s1}["a"] + {value}["c"]).cast("DOUBLE")',
            ),
        }
        actual_named = _build_transformation_from_tree(
            _build_nested_struct_tree(named_transformations),
        )
        assert str(actual_named) == (
            """lambda x: [f.transform(x['s1'], lambda x: f.struct([x['a'].alias('a'), """
            """f.transform_values(f.transform_keys(x['m1'], lambda k, v: """
            """f.struct([(x["a"] + k["b"]).cast("DOUBLE").alias('b')]))),lambda k, v: """
            """f.struct([(x["a"] + v["c"]).cast("DOUBLE").alias('c')])).alias('m1')])).alias('s1')]"""
        )


class TestResolveNestedFields:
    def test_with_error(self):
        """
        GIVEN a DataFrame with a simple value
        WHEN we use resolve_nested_columns on it with an incorrect expression
        THEN an AnalysisException should be raised
        """
        transformation = {"a!b": None}
        with pytest.raises(spark_frame.exceptions.AnalysisException) as e:
            resolve_nested_fields(transformation)
        assert "Invalid field name 'a!b'" in str(e.value)

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
            |""",
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
            |""",
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
            |""",
        )
        named_transformations = {
            "a": PrintableFunction(lambda s: f.col("a"), lambda s: 'f.col("a")'),
        }
        transformations = {"a": f.col("a")}
        expected = strip_margin(
            """
            |+---+
            ||  a|
            |+---+
            ||  1|
            |+---+
            |""",
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
            |""",
        )
        named_transformations = {
            "a": PrintableFunction(
                lambda s: f.col("a").alias("other_alias"),
                lambda s: 'f.col("a").alias("other_alias")',
            ),
        }
        transformations = {"a": f.col("a").alias("other_alias")}
        expected = strip_margin(
            """
            |+---+
            ||  a|
            |+---+
            ||  1|
            |+---+
            |""",
        )
        assert show_string(df.select(*resolve_nested_fields(named_transformations))) == expected
        assert show_string(df.select(*resolve_nested_fields(transformations))) == expected

    def test_value_with_get_expr(self, spark: SparkSession):
        """
        GIVEN a DataFrame with a simple value
        WHEN we use resolve_nested_columns on it with a lambda function that accesses the root level
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
            |""",
        )
        named_transformations = {
            "a": PrintableFunction(lambda r: r["a"], lambda r: f'{r}["a"]'),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected = strip_margin(
            """
            |+---+
            ||  a|
            |+---+
            ||  1|
            |+---+
            |""",
        )
        assert (
            show_string(
                df.select(
                    *resolve_nested_fields(named_transformations, starting_level=df),
                ),
            )
            == expected
        )
        assert (
            show_string(
                df.select(*resolve_nested_fields(transformations, starting_level=df)),
            )
            == expected
        )

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
            |""",
        )
        named_transformations = {
            "s.a": PrintableFunction(
                lambda s: f.col("s.a").cast("DOUBLE"),
                lambda s: """f.col("s.a").cast("DOUBLE")""",
            ),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected = strip_margin(
            """
            |+-----+
            ||    s|
            |+-----+
            ||{2.0}|
            |+-----+
            |""",
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
            |""",
        )
        named_transformations = {
            "s.a": PrintableFunction(
                lambda s: f.col("s.a").cast("DOUBLE"),
                lambda s: """f.col("s.a").cast("DOUBLE")""",
            ),
        }
        transformations = {"s.a": f.col("s.a").cast("DOUBLE")}
        expected = strip_margin(
            """
            |+-----+
            ||    s|
            |+-----+
            ||{2.0}|
            |+-----+
            |""",
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
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+------+
            ||     e|
            |+------+
            ||[2, 3]|
            |+------+
            |""",
        )
        named_transformations = {
            "e!": PrintableFunction(
                lambda e: e.cast("DOUBLE"),
                lambda e: f'{e}.cast("DOUBLE")',
            ),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- e: array (nullable = false)
            | |    |-- element: double (containsNull = false)
            |""",
        )
        expected = strip_margin(
            """
            |+----------+
            ||         e|
            |+----------+
            ||[2.0, 3.0]|
            |+----------+
            |""",
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_array_with_none(self, spark: SparkSession):
        """
        GIVEN a DataFrame with an array
        WHEN we use resolve_nested_columns on it with a transformation being None
        THEN the transformation should work
        """
        df = spark.sql("SELECT ARRAY(2, 3) as e")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- e: array (nullable = false)
            | |    |-- element: integer (containsNull = false)
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+------+
            ||     e|
            |+------+
            ||[2, 3]|
            |+------+
            |""",
        )
        named_transformations = {"e!": None}
        expected_schema = strip_margin(
            """
            |root
            | |-- e: array (nullable = false)
            | |    |-- element: integer (containsNull = false)
            |""",
        )
        expected = strip_margin(
            """
            |+------+
            ||     e|
            |+------+
            ||[2, 3]|
            |+------+
            |""",
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected

    def test_array_with_str_expr(self, spark: SparkSession):
        """
        GIVEN a DataFrame with an array
        WHEN we use resolve_nested_columns on it with a transformation being a string expression
        THEN the transformation should work
        """
        df = spark.sql("SELECT 1 as id, ARRAY(2, 3) as e")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- id: integer (nullable = false)
            | |-- e: array (nullable = false)
            | |    |-- element: integer (containsNull = false)
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+---+------+
            || id|     e|
            |+---+------+
            ||  1|[2, 3]|
            |+---+------+
            |""",
        )
        named_transformations = {"id": None, "e!": "id"}
        expected_schema = strip_margin(
            """
            |root
            | |-- id: integer (nullable = false)
            | |-- e: array (nullable = false)
            | |    |-- element: integer (containsNull = false)
            |""",
        )
        expected = strip_margin(
            """
            |+---+------+
            || id|     e|
            |+---+------+
            ||  1|[1, 1]|
            |+---+------+
            |""",
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected

    def test_array_with_col_expr(self, spark: SparkSession):
        """
        GIVEN a DataFrame with an array
        WHEN we use resolve_nested_columns on it with a transformation being a Column expression
        THEN the transformation should work
        """
        df = spark.sql("SELECT ARRAY(2, 3) as e")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- e: array (nullable = false)
            | |    |-- element: integer (containsNull = false)
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+------+
            ||     e|
            |+------+
            ||[2, 3]|
            |+------+
            |""",
        )
        named_transformations = {"e!": f.lit(1)}
        expected_schema = strip_margin(
            """
            |root
            | |-- e: array (nullable = false)
            | |    |-- element: integer (containsNull = false)
            |""",
        )
        expected = strip_margin(
            """
            |+------+
            ||     e|
            |+------+
            ||[1, 1]|
            |+------+
            |""",
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected

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
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+-----+
            ||    s|
            |+-----+
            ||[{2}]|
            |+-----+
            |""",
        )
        named_transformations = {
            "s!.a": PrintableFunction(
                lambda s: s["a"].cast("DOUBLE"),
                lambda s: f'{s}["a"].cast("DOUBLE")',
            ),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- a: double (nullable = false)
            |""",
        )
        expected = strip_margin(
            """
            |+-------+
            ||      s|
            |+-------+
            ||[{2.0}]|
            |+-------+
            |""",
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
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+-----+
            ||   s1|
            |+-----+
            ||{{2}}|
            |+-----+
            |""",
        )
        named_transformations = {
            "s1.s2.a": PrintableFunction(
                lambda s: f.col("s1")["s2"]["a"].cast("DOUBLE"),
                lambda s: 'f.col("s1")["s2"]["a"].cast("DOUBLE")',
            ),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1: struct (nullable = false)
            | |    |-- s2: struct (nullable = false)
            | |    |    |-- a: double (nullable = false)
            |""",
        )
        expected = strip_margin(
            """
            |+-------+
            ||     s1|
            |+-------+
            ||{{2.0}}|
            |+-------+
            |""",
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
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+--------+
            ||      s1|
            |+--------+
            ||{[2, 3]}|
            |+--------+
            |""",
        )
        named_transformations = {
            "s1.e!": PrintableFunction(
                lambda e: e.cast("DOUBLE"),
                lambda e: f'{e}.cast("DOUBLE")',
            ),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1: struct (nullable = false)
            | |    |-- e: array (nullable = false)
            | |    |    |-- element: double (containsNull = false)
            |""",
        )
        expected = strip_margin(
            """
            |+------------+
            ||          s1|
            |+------------+
            ||{[2.0, 3.0]}|
            |+------------+
            |""",
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
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+-------+
            ||     s1|
            |+-------+
            ||{[{2}]}|
            |+-------+
            |""",
        )
        named_transformations = {
            "s1.s2!.a": PrintableFunction(
                lambda s: s["a"].cast("DOUBLE"),
                lambda s: f'{s}["a"].cast("DOUBLE")',
            ),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1: struct (nullable = false)
            | |    |-- s2: array (nullable = false)
            | |    |    |-- element: struct (containsNull = false)
            | |    |    |    |-- a: double (nullable = false)
            |""",
        )
        expected = strip_margin(
            """
            |+---------+
            ||       s1|
            |+---------+
            ||{[{2.0}]}|
            |+---------+
            |""",
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
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+-----+
            ||    e|
            |+-----+
            ||[[1]]|
            |+-----+
            |""",
        )
        named_transformations = {
            "e!!": PrintableFunction(
                lambda e: e.cast("DOUBLE"),
                lambda e: f'{e}.cast("DOUBLE")',
            ),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- e: array (nullable = false)
            | |    |-- element: array (containsNull = false)
            | |    |    |-- element: double (containsNull = false)
            |""",
        )
        expected = strip_margin(
            """
            |+-------+
            ||      e|
            |+-------+
            ||[[1.0]]|
            |+-------+
            |""",
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
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+-------+
            ||      s|
            |+-------+
            ||[[{1}]]|
            |+-------+
            |""",
        )
        named_transformations = {
            "s!!.a": PrintableFunction(
                lambda s: s["a"].cast("DOUBLE"),
                lambda s: f'{s}["a"].cast("DOUBLE")',
            ),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s: array (nullable = false)
            | |    |-- element: array (containsNull = false)
            | |    |    |-- element: struct (containsNull = false)
            | |    |    |    |-- a: double (nullable = false)
            |""",
        )
        expected = strip_margin(
            """
            |+---------+
            ||        s|
            |+---------+
            ||[[{1.0}]]|
            |+---------+
            |""",
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_struct_in_array_struct(self, spark: SparkSession):
        """
        GIVEN a DataFrame with a struct inside a struct inside an array
        WHEN we use resolve_nested_columns with a transformation that access an element from the outermost struct
        THEN the transformation should work
        """
        df = spark.sql("SELECT ARRAY(STRUCT(1 as a, STRUCT(2 as b) as s2)) as s1")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- s1: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- a: integer (nullable = false)
            | |    |    |-- s2: struct (nullable = false)
            | |    |    |    |-- b: integer (nullable = false)
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+----------+
            ||        s1|
            |+----------+
            ||[{1, {2}}]|
            |+----------+
            |""",
        )
        # Here, the lambda function is applied to the elements of `s1`, not `s2`
        named_transformations = {
            "s1!.a": PrintableFunction(lambda s1: s1["a"], lambda s1: f'{s1}["a"]'),
            "s1!.s2.b": PrintableFunction(
                lambda s1: s1["a"].cast("DOUBLE"),
                lambda s1: f'{s1}["a"].cast("DOUBLE")',
            ),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- a: integer (nullable = false)
            | |    |    |-- s2: struct (nullable = false)
            | |    |    |    |-- b: double (nullable = false)
            |""",
        )
        expected = strip_margin(
            """
            |+------------+
            ||          s1|
            |+------------+
            ||[{1, {1.0}}]|
            |+------------+
            |""",
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
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+----------+
            ||        s1|
            |+----------+
            ||[{[2, 3]}]|
            |+----------+
            |""",
        )
        named_transformations = {
            "s1!.e!": PrintableFunction(
                lambda e: e.cast("DOUBLE"),
                lambda e: f'{e}.cast("DOUBLE")',
            ),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- e: array (nullable = false)
            | |    |    |    |-- element: double (containsNull = false)
            |""",
        )
        expected = strip_margin(
            """
            |+--------------+
            ||            s1|
            |+--------------+
            ||[{[2.0, 3.0]}]|
            |+--------------+
            |""",
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
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+---------+
            ||       s1|
            |+---------+
            ||[{[{2}]}]|
            |+---------+
            |""",
        )
        named_transformations = {
            "s1!.s2!.a": PrintableFunction(
                lambda s: s["a"].cast("DOUBLE"),
                lambda s: f'{s}["a"].cast("DOUBLE")',
            ),
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
            |""",
        )
        expected = strip_margin(
            """
            |+-----------+
            ||         s1|
            |+-----------+
            ||[{[{2.0}]}]|
            |+-----------+
            |""",
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_array_struct_in_array_struct_with_transformation_using_field_from_first_array(
        self,
        spark: SparkSession,
    ):
        """
        GIVEN a DataFrame with an array<struct> inside another array<struct>
        WHEN we use resolve_nested_columns on it with a transformation that uses data from the first array
        THEN the transformation should work
        """
        df = spark.sql(
            "SELECT ARRAY(STRUCT(1 as a, ARRAY(STRUCT(2 as b)) as s2)) as s1",
        )
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- s1: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- a: integer (nullable = false)
            | |    |    |-- s2: array (nullable = false)
            | |    |    |    |-- element: struct (containsNull = false)
            | |    |    |    |    |-- b: integer (nullable = false)
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+------------+
            ||          s1|
            |+------------+
            ||[{1, [{2}]}]|
            |+------------+
            |""",
        )
        named_transformations = {
            "s1!.a": PrintableFunction(lambda s1: s1["a"], lambda s1: f'{s1}["a"]'),
            "s1!.s2!.b": PrintableFunction(
                lambda s1, s2: (s1["a"] + s2["b"]).cast("DOUBLE"),
                lambda s1, s2: f'({s1}["a"]+{s2}["b"]).cast("DOUBLE")',
            ),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1: array (nullable = false)
            | |    |-- element: struct (containsNull = false)
            | |    |    |-- a: integer (nullable = false)
            | |    |    |-- s2: array (nullable = false)
            | |    |    |    |-- element: struct (containsNull = false)
            | |    |    |    |    |-- b: double (nullable = false)
            |""",
        )
        expected = strip_margin(
            """
            |+--------------+
            ||            s1|
            |+--------------+
            ||[{1, [{3.0}]}]|
            |+--------------+
            |""",
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual_named.show()
        actual = df.select(*resolve_nested_fields(transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_struct_in_array_struct_in_array_struct(self, spark: SparkSession):
        """
        GIVEN a DataFrame with a struct inside an array<struct> inside an array<struct>
        WHEN we use resolve_nested_columns with a transformation that access an element from the outermost struct
        THEN the transformation should work
        """
        df = spark.sql(
            """SELECT ARRAY(STRUCT(ARRAY(STRUCT(1 as a, STRUCT(2 as b) as s3)) as s2)) as s1""",
        )
        assert nested.schema_string(df) == strip_margin(
            """
            |root
            | |-- s1!.s2!.a: integer (nullable = false)
            | |-- s1!.s2!.s3.b: integer (nullable = false)
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+--------------+
            ||            s1|
            |+--------------+
            ||[{[{1, {2}}]}]|
            |+--------------+
            |""",
        )
        # Here, the lambda function is applied to the elements of `s2`, not `s3`
        named_transformations = {
            "s1!.s2!.a": PrintableFunction(lambda s2: s2["a"], lambda s2: f'{s2}["a"]'),
            "s1!.s2!.s3.b": PrintableFunction(
                lambda s2: s2["a"].cast("DOUBLE"),
                lambda s2: f'{s2}["a"].cast("DOUBLE")',
            ),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1!.s2!.a: integer (nullable = false)
            | |-- s1!.s2!.s3.b: double (nullable = false)
            |""",
        )
        expected = strip_margin(
            """
            |+----------------+
            ||              s1|
            |+----------------+
            ||[{[{1, {1.0}}]}]|
            |+----------------+
            |""",
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert nested.schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert nested.schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_struct_in_struct_in_array_struct_in_struct_in_array_struct(
        self,
        spark: SparkSession,
    ):
        """
        GIVEN a DataFrame with a struct inside a struct inside an array<struct> inside a struct inside an array<struct>
        WHEN we use resolve_nested_columns with a transformation that access an element from the outermost struct
        THEN the transformation should work
        """
        df = spark.sql(
            """SELECT
            ARRAY(STRUCT(STRUCT(
                ARRAY(STRUCT(STRUCT(1 as a, STRUCT(2 as b) as s5) as s4)) as s3
            ) as s2)) as s1
        """,
        )
        assert nested.schema_string(df) == strip_margin(
            """
            |root
            | |-- s1!.s2.s3!.s4.a: integer (nullable = false)
            | |-- s1!.s2.s3!.s4.s5.b: integer (nullable = false)
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+------------------+
            ||                s1|
            |+------------------+
            ||[{{[{{1, {2}}}]}}]|
            |+------------------+
            |""",
        )
        # Here, the lambda function is applied to the elements of `s3`, not `s4` or `s5`
        named_transformations = {
            "s1!.s2.s3!.s4.a": PrintableFunction(
                lambda s3: s3["s4"]["a"],
                lambda s3: f'{s3}["s4"]["a"]',
            ),
            "s1!.s2.s3!.s4.s5.b": PrintableFunction(
                lambda s3: s3["s4"]["a"].cast("DOUBLE"),
                lambda s3: f'{s3}["s4"]["a"].cast("DOUBLE")',
            ),
        }
        transformations = replace_named_functions_with_functions(named_transformations)
        expected_schema = strip_margin(
            """
            |root
            | |-- s1!.s2.s3!.s4.a: integer (nullable = false)
            | |-- s1!.s2.s3!.s4.s5.b: double (nullable = false)
            |""",
        )
        expected = strip_margin(
            """
            |+--------------------+
            ||                  s1|
            |+--------------------+
            ||[{{[{{1, {1.0}}}]}}]|
            |+--------------------+
            |""",
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        actual = df.select(*resolve_nested_fields(transformations))
        assert nested.schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected
        assert nested.schema_string(actual) == expected_schema
        assert show_string(actual) == expected

    def test_struct_in_struct_in_array_struct_in_struct_in_array_struct_with_none(
        self,
        spark: SparkSession,
    ):
        """
        GIVEN a DataFrame with a struct inside a struct inside an array<struct> inside a struct inside an array<struct>
        WHEN we use resolve_nested_columns with a None transformation
        THEN the transformation should work
        """
        df = spark.sql(
            """SELECT
            ARRAY(STRUCT(STRUCT(
                ARRAY(STRUCT(STRUCT(1 as a, STRUCT(2 as b) as s5) as s4)) as s3
            ) as s2)) as s1
        """,
        )
        assert nested.schema_string(df) == strip_margin(
            """
            |root
            | |-- s1!.s2.s3!.s4.a: integer (nullable = false)
            | |-- s1!.s2.s3!.s4.s5.b: integer (nullable = false)
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+------------------+
            ||                s1|
            |+------------------+
            ||[{{[{{1, {2}}}]}}]|
            |+------------------+
            |""",
        )
        # Here, the lambda function is applied to the elements of `s3`, not `s4` or `s5`
        named_transformations = {"s1!.s2.s3!.s4.a": None, "s1!.s2.s3!.s4.s5.b": None}
        expected_schema = strip_margin(
            """
            |root
            | |-- s1!.s2.s3!.s4.a: integer (nullable = false)
            | |-- s1!.s2.s3!.s4.s5.b: integer (nullable = false)
            |""",
        )
        expected = strip_margin(
            """
            |+------------------+
            ||                s1|
            |+------------------+
            ||[{{[{{1, {2}}}]}}]|
            |+------------------+
            |""",
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        assert nested.schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected

    def test_struct_in_map_values(self, spark: SparkSession):
        """
        GIVEN a DataFrame with a struct inside map values
        WHEN we use resolve_nested_columns on it
        THEN the transformation should work
        """
        df = spark.sql("""SELECT MAP("a", STRUCT(2 as a)) as m1""")
        assert schema_string(df) == strip_margin(
            """
            |root
            | |-- m1: map (nullable = false)
            | |    |-- key: string
            | |    |-- value: struct (valueContainsNull = false)
            | |    |    |-- a: integer (nullable = false)
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+----------+
            ||        m1|
            |+----------+
            ||{a -> {2}}|
            |+----------+
            |""",
        )
        named_transformations = {
            "m1%key": identity,
            "m1%value.a": PrintableFunction(
                lambda value: value["a"].cast("DOUBLE"),
                lambda s: f'{s}["a"].cast("DOUBLE")',
            ),
        }
        expected_schema = strip_margin(
            """
            |root
            | |-- m1: map (nullable = false)
            | |    |-- key: string
            | |    |-- value: struct (valueContainsNull = false)
            | |    |    |-- a: double (nullable = false)
            |""",
        )
        expected = strip_margin(
            """
            |+------------+
            ||          m1|
            |+------------+
            ||{a -> {2.0}}|
            |+------------+
            |""",
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        assert schema_string(actual_named) == expected_schema
        assert show_string(actual_named) == expected

    def test_struct_in_map_values_in_array_struct_with_transformation_using_field_from_first_array(
        self,
        spark: SparkSession,
    ):
        """
        GIVEN a DataFrame with a struct inside map values inside an array<struct>
        WHEN we use resolve_nested_columns on it with a transformation that uses data from the first array
        THEN the transformation should work
        """
        df = spark.sql(
            """SELECT ARRAY(STRUCT(1 as a, MAP(STRUCT(2 as b), STRUCT(3 as c)) as m1)) as s1""",
        )
        assert nested.schema_string(df) == strip_margin(
            """
            |root
            | |-- s1!.a: integer (nullable = false)
            | |-- s1!.m1%key.b: integer (nullable = false)
            | |-- s1!.m1%value.c: integer (nullable = false)
            |""",
        )
        assert show_string(df) == strip_margin(
            """
            |+-------------------+
            ||                 s1|
            |+-------------------+
            ||[{1, {{2} -> {3}}}]|
            |+-------------------+
            |""",
        )
        named_transformations = {
            "s1!.a": None,
            "s1!.m1%key.b": PrintableFunction(
                lambda s1, key: (s1["a"] + key["b"]).cast("DOUBLE"),
                lambda s1, key: f'({s1}["a"] + {key}["b"]).cast("DOUBLE")',
            ),
            "s1!.m1%value.c": PrintableFunction(
                lambda s1, value: (s1["a"] + value["c"]).cast("DOUBLE"),
                lambda s1, value: f'({s1}["a"] + {value}["c"]).cast("DOUBLE")',
            ),
        }
        expected_schema = strip_margin(
            """
            |root
            | |-- s1!.a: integer (nullable = false)
            | |-- s1!.m1%key.b: double (nullable = false)
            | |-- s1!.m1%value.c: double (nullable = false)
            |""",
        )
        expected = strip_margin(
            """
            |+-----------------------+
            ||s1                     |
            |+-----------------------+
            ||[{1, {{3.0} -> {4.0}}}]|
            |+-----------------------+
            |""",
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        assert nested.schema_string(actual_named) == expected_schema
        assert show_string(actual_named, truncate=False) == expected

    def test_struct_in_array_struct_in_map_values_with_transformation_using_field_from_first_map(
        self,
        spark: SparkSession,
    ):
        """
        GIVEN a DataFrame with a struct inside an array<struct> inside map values
        WHEN we use resolve_nested_columns on it with a transformation that uses data from the first array
        THEN the transformation should work
        """
        df = spark.sql(
            """SELECT MAP(
            STRUCT(1 as a, ARRAY(STRUCT(2 as b)) as s1),
            STRUCT(3 as c, ARRAY(STRUCT(2 as d)) as s2)
        ) as m1
        """,
        )
        assert nested.schema_string(df) == strip_margin(
            """
            |root
            | |-- m1%key.a: integer (nullable = false)
            | |-- m1%key.s1!.b: integer (nullable = false)
            | |-- m1%value.c: integer (nullable = false)
            | |-- m1%value.s2!.d: integer (nullable = false)
            |""",
        )
        assert show_string(df, truncate=False) == strip_margin(
            """
            |+--------------------------+
            ||m1                        |
            |+--------------------------+
            ||{{1, [{2}]} -> {3, [{2}]}}|
            |+--------------------------+
            |""",
        )
        named_transformations = {
            "m1%key.a": None,
            "m1%key.s1!.b": PrintableFunction(
                lambda key, s1: (key["a"] + s1["b"]).cast("DOUBLE"),
                lambda key, s1: f'({key}["a"] + {s1}["b"]).cast("DOUBLE")',
            ),
            "m1%value.c": None,
            "m1%value.s2!.d": PrintableFunction(
                lambda value, s2: (value["c"] + s2["d"]).cast("DOUBLE"),
                lambda value, s2: f'({value}["c"] + {s2}["d"]).cast("DOUBLE")',
            ),
        }
        expected_schema = strip_margin(
            """
            |root
            | |-- m1%key.a: integer (nullable = false)
            | |-- m1%key.s1!.b: double (nullable = false)
            | |-- m1%value.c: integer (nullable = false)
            | |-- m1%value.s2!.d: double (nullable = false)
            |""",
        )
        expected = strip_margin(
            """
            |+------------------------------+
            ||m1                            |
            |+------------------------------+
            ||{{1, [{3.0}]} -> {3, [{5.0}]}}|
            |+------------------------------+
            |""",
        )
        actual_named = df.select(*resolve_nested_fields(named_transformations))
        assert nested.schema_string(actual_named) == expected_schema
        assert show_string(actual_named, truncate=False) == expected
