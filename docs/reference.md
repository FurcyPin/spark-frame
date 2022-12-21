## spark_frame.functions

Like with [pyspark.sql.functions](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html), 
the methods in this module all return [Column][pyspark.sql.Column] expressions and can be used to build operations
on Spark [DataFrames][pyspark.sql.DataFrame] using `select`, `withColumn`, etc.

---

### ::: spark_frame.functions.empty_array
---

### ::: spark_frame.functions.generic_struct
---

### ::: spark_frame.functions.nullable
---


## spark_frame.graph

This module contains implementations of graph algorithms and related methods.

---

### ::: spark_frame.graph_impl.ascending_forest_traversal.ascending_forest_traversal
---


## spark_frame.transformations

Unlike those in [spark_frame.functions](#spark_framefunctions), the methods in this module all take at least one
[DataFrame][pyspark.sql.DataFrame] as argument and return a new transformed DataFrame.
These methods generally offer _higher order_ transformation that requires to inspect the schema or event the content
of the input DataFrame(s) before generating the next transformation. Those are typically generic operations 
that _cannot_ be implemented with one single SQL query.

!!! tip

    Since Spark 3.3.0, all transformations can be inlined using 
    [DataFrame.transform][pyspark.sql.DataFrame.transform], like this:

    ```python
    df.transform(flatten).withColumn(
        "base_stats.Total",
        f.col("`base_stats.Attack`") + f.col("`base_stats.Defense`") + f.col("`base_stats.HP`") +
        f.col("`base_stats.Sp Attack`") + f.col("`base_stats.Sp Defense`") + f.col("`base_stats.Speed`")
    ).transform(unflatten).show(vertical=True, truncate=False)
    ```
    *This example is taken*

---


### ::: spark_frame.transformations_impl.convert_all_maps_to_arrays.convert_all_maps_to_arrays
---

### ::: spark_frame.transformations_impl.harmonize_dataframes.harmonize_dataframes
---

### ::: spark_frame.transformations_impl.parse_json_columns.parse_json_columns
---

### ::: spark_frame.transformations_impl.sort_all_arrays.sort_all_arrays
---

### ::: spark_frame.transformations_impl.unflatten.unflatten
---

### ::: spark_frame.transformations_impl.unpivot.unpivot
---

### ::: spark_frame.transformations_impl.with_generic_typed_struct.with_generic_typed_struct
---


## spark_frame.nested

### Please read this before using the `spark_frame.nested` module

The `spark_frame.nested` module contains several methods that make the manipulation of deeply nested data structures 
much easier. Before diving into it, it is important to explicit the concept of `Field` in the context of this library.

::: spark_frame.examples.reference_nested.fields
    options:
        show_root_toc_entry: false
        show_root_heading: false
        show_source: false
---

### ::: spark_frame.nested_impl.print_schema.print_schema
---

### ::: spark_frame.nested_impl.select_impl.select
---

### ::: spark_frame.nested_impl.schema_string.schema_string
---

### ::: spark_frame.nested_impl.unnest_all_fields.unnest_all_fields
---

### ::: spark_frame.nested_impl.unnest_field.unnest_field
---

### ::: spark_frame.nested_impl.with_fields.with_fields
---

## spark_frame.schema_utils

This module contains methods useful for manipulating DataFrame schemas.

---

### ::: spark_frame.schema_utils.schema_from_json
---
### ::: spark_frame.schema_utils.schema_from_simple_string
---
### ::: spark_frame.schema_utils.schema_to_json
---
### ::: spark_frame.schema_utils.schema_to_pretty_json
---
### ::: spark_frame.schema_utils.schema_to_simple_string
---
