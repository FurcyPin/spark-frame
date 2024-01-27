## Transforming nested fields

!!! warning
    The use case presented in this page is deprecated, but is kept to illustrate what flatten/unflatten can do.
    The [spark_frame.nested](/spark-frame/reference/nested) module is much more powerful for manipulating nested data, 
    because unlike flatten/unflatten, it does work with arrays. We recommend checking 
    [this use-case](/use_cases/working_with_nested_data) to see the [spark_frame.nested](/spark-frame/reference/nested)
    module in action.

::: spark_frame.examples.flatten_unflatten.transform_nested_fields
    options:
        show_root_toc_entry: false
        show_root_heading: false
        show_source: false

**Methods used in this example**

??? abstract "transformations.flatten"
    ::: spark_frame.transformations.flatten
        options:
            show_root_heading: false
            show_root_toc_entry: false

??? abstract "transformations.unflatten"
    ::: spark_frame.transformations.unflatten
        options:
            show_root_heading: false
            show_root_toc_entry: false

