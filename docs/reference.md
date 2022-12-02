## spark_frame.functions

Like with [pyspark.sql.functions](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html), 
the methods in this module all return [Column](pyspark.sql.Column) expressions and can be used to build operations
on Spark [DataFrames](pyspark.sql.DataFrame) using `select`, `withColumn`, etc.

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
[DataFrame](pyspark.sql.DataFrame) as argument and return a new transformed DataFrame.
These methods generally offer _higher order_ transformation that requires to inspect the schema or event the content
of the input DataFrame(s) before generating the next transformation. Those are typically generic operations 
that _cannot_ be implemented with one single SQL query.

---


### ::: spark_frame.transformations_impl.flatten.flatten
---

### ::: spark_frame.transformations_impl.parse_json_columns.parse_json_columns
---

### ::: spark_frame.transformations_impl.unflatten.unflatten
---

### ::: spark_frame.transformations_impl.unpivot.unpivot
---

### ::: spark_frame.transformations_impl.with_generic_typed_struct.with_generic_typed_struct
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
