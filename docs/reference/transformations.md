Unlike those in [spark_frame.functions](/spark-frame/reference/functions), the methods in this module all take at least one
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

### ::: spark_frame.transformations_impl.analyze.analyze
---
### ::: spark_frame.transformations_impl.convert_all_maps_to_arrays.convert_all_maps_to_arrays
---
### ::: spark_frame.transformations_impl.flatten.flatten
---
### ::: spark_frame.transformations_impl.flatten_all_arrays.flatten_all_arrays
---
### ::: spark_frame.transformations_impl.harmonize_dataframes.harmonize_dataframes
---
### ::: spark_frame.transformations_impl.parse_json_columns.parse_json_columns
---
### ::: spark_frame.transformations_impl.sort_all_arrays.sort_all_arrays
---
### ::: spark_frame.transformations_impl.transform_all_field_names.transform_all_field_names
---
### ::: spark_frame.transformations_impl.transform_all_fields.transform_all_fields
---
### ::: spark_frame.transformations_impl.unflatten.unflatten
---
### ::: spark_frame.transformations_impl.union_dataframes.union_dataframes
---
### ::: spark_frame.transformations_impl.unpivot.unpivot
---
### ::: spark_frame.transformations_impl.with_generic_typed_struct.with_generic_typed_struct
---
