This module contains a the method [`compare_dataframes`][spark_frame.data_diff.compare_dataframes] 
which is used to compare two DataFrames.
It generates a [`DiffResult`][spark_frame.data_diff.diff_result.DiffResult] object, which
can be used to display the results on `stdout` or even be exported as an interactive HTML report file. 
For more advanced use cases, the underlying results can also be accessed as DataFrames.


---

### ::: spark_frame.data_diff.compare_dataframes

---


### ::: spark_frame.data_diff.DiffResult
    options:
        members:
            - diff_df_shards
            - display
            - export_to_html
            - get_diff_per_col_df


### ::: spark_frame.data_diff.DiffFormatOptions

