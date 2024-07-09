# Spark-frame

[![PyPI version](https://badge.fury.io/py/spark-frame.svg)](https://badge.fury.io/py/spark-frame)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spark-frame.svg)](https://pypi.org/project/spark-frame/)
[![GitHub Build](https://img.shields.io/github/actions/workflow/status/FurcyPin/spark-frame/build_and_validate.yml?branch=main)](https://github.com/FurcyPin/spark-frame/actions)
[![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=FurcyPin_spark-frame&metric=coverage)](https://sonarcloud.io/component_measures?id=FurcyPin_spark-frame&metric=coverage&view=list)
[![SonarCloud Bugs](https://sonarcloud.io/api/project_badges/measure?project=FurcyPin_spark-frame&metric=bugs)](https://sonarcloud.io/component_measures?metric=reliability_rating&view=list&id=FurcyPin_spark-frame)
[![SonarCloud Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=FurcyPin_spark-frame&metric=vulnerabilities)](https://sonarcloud.io/component_measures?metric=security_rating&view=list&id=FurcyPin_spark-frame)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/spark-frame)](https://pypi.org/project/spark-frame/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## What is it ?

**[Spark-frame](https://furcypin.github.io/spark-frame/) is a library that super-charges your Spark DataFrames!**

It brings several utility methods and transformation functions for PySpark DataFrames.

Here is a quick list of the most exciting features :sunglasses:

- `spark_frame.data_diff.compare_dataframes`: compare two SQL tables or DataFrames and generate an HTML report 
  to view the result. And yes, this is completely free, open source, and it works even with 
  complex data structures ! It also detects column reordering and can handle type changes.
  [Go check it out :exclamation:](https://furcypin.github.io/spark-frame/use_cases/comparing_dataframes/)
- `spark_frame.nested`: Did you ever thought manipulating complex data structures in SQL or Spark was a 
  nightmare :jack_o_lantern: ? You just found the solution ! The `nested` library  makes those manipulations much 
  cleaner and simpler. 
  [Get started over there :rocket:](https://furcypin.github.io/spark-frame/use_cases/working_with_nested_data/)
- `spark_frame.transformations`: A wide collection of generic dataframe transformations.
    - Ever wanted to apply a transformation to every field of a DataFrame depending on it's name or type ? 
      [Easy as pie :cake:](https://furcypin.github.io/spark-frame/reference/transformations/#spark_frame.transformations_impl.transform_all_fields.transform_all_fields)
    - Ever wanted to rename every field of a DataFrame, including the deeply nested ones ? 
      [Done: :ok_hand:](https://furcypin.github.io/spark-frame/reference/transformations/#spark_frame.transformations_impl.transform_all_field_names.transform_all_field_names)
    - Ever wanted to analyze the content of a DataFrame, 
      but [`DataFrame.describe()`](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.describe.html)
      does not work with complex data types ? 
      [You're welcome :pray:](https://furcypin.github.io/spark-frame/reference/transformations/#spark_frame.transformations_impl.analyze.analyze)
- `spark_frame.schema_utils`: Need to dump the schema of a DataFrame somewhere to be able to load it later ?
     [We got you covered :thumbsup:](https://furcypin.github.io/spark-frame/reference/schema_utils/)
- `spark_frame.graph.ascending_forest_traversal`: Need an algorithm that takes the adjacency matrix of a 
 tree :deciduous_tree: (or forest) graph and associates each node to their corresponding root node ?
 But that other algorithm you tried went into an infinite loop ∞ because your graph isn't really a tree 
 and occasionally contains cycles ? 
 [Try this :evergreen_tree:](https://furcypin.github.io/spark-frame/reference/graph/#spark_frame.graph_impl.ascending_forest_traversal.ascending_forest_traversal)


## Getting Started

Visit the official Spark-frame website [documentation](https://furcypin.github.io/spark-frame/) 
for [use cases examples](https://furcypin.github.io/spark-frame/use_cases/intro/) 
and [reference](https://furcypin.github.io/spark-frame/reference/functions/).

## Installation

[spark-frame is available on PyPi](https://pypi.org/project/spark-frame/).

```bash
pip install spark-frame
```

## Compatibilities and requirements


This library does not depend on any other library.
**Pyspark must be installed separately to use it.**
It is compatible with the following versions:

- Python: requires 3.8.1 or higher (tested against Python 3.8, 3.9, 3.10, 3.11 and 3.12)
- Pyspark: requires 3.3.0 or higher (tested against PySpark 3.3, 3.4 and 3.5)

This library is tested against Windows, Mac and Linux.

However, testing for the following combinations has been disabled, because they are failing:

- PySpark 3.3 with Python >= 3.11
- PySpark 3.4 with Python >= 3.12
- PySpark 3.5 with Python 3.12 on Windows


**Some features require extra libraries to be installed alongside this project.**
**We chose to not include them as direct dependencies for security and flexibility reasons.**
**This way, users who are not using these features don't need to worry about these dependencies.**

| feature                                    |  Method                      | spark-frame's <br> version |               dependency required |
|--------------------------------------------|------------------------------|----------------------------|----------------------------------:|
| Generating HTML <br> reports for data diff |  `DiffResult.export_to_html` | >= 0.5.0                   | data-diff-viewer==0.3.* (>=0.3.2) |
| Generating HTML <br> reports for data diff |  `DiffResult.export_to_html` | 0.4.*                      |           data-diff-viewer==0.2.* |
| Generating HTML <br> reports for data diff |  `DiffResult.export_to_html` | < 0.4                      |                            jinja2 |

_Since version 0.4, the code used to generate HTML diff reports has been moved to 
[data-diff-viewer](https://github.com/FurcyPin/data-diff-viewer) from the same author. 
It comes with a dependency to [duckdb](https://github.com/duckdb/duckdb), 
which is used to store the diff results and embed them in the HTML page._


# Genesis of the project

These methods were initially part of the [karadoc](https://github.com/FurcyPin/karadoc) project 
used at [Younited](https://medium.com/younited-tech-blog), but they were fully independent from karadoc, 
so it made more sense to keep them as a standalone library.

Several of these methods were my initial inspiration to make the cousin project
[bigquery-frame](https://github.com/FurcyPin/bigquery-frame), which was first made to illustrate
this [blog article](https://medium.com/towards-data-science/sql-jinja-is-not-enough-why-we-need-dataframes-4d71a191936d).
This is why you will find similar methods in both `spark_frame` and `bigquery_frame`, 
except the former runs on PySpark while the latter runs on BigQuery (obviously).
I try to keep both projects consistent together, and will eventually port new developments made on 
one project to the other one.


# Changelog

# v0.5.1

**Bugfixes**

- data-diff:
  - Fix export of HTML report not working in cluster mode.

# v0.5.0

**New features:**

- data-diff:
  - Full sample rows in data-diff: in the data-diff HTML report, you can now click on a most frequent 
    value or change for a column and it will display the full content of a row where this change happens.
    

**Breaking Changes:**

- data-diff:
  - The names of the keys of the `DiffResult.diff_df_shards` dict have changed: 
    All keys except the root key (`""`) have been appended a REPETITION_MARKER (`"!"`).
    This will make future manipulations easier. This should not impact users as it is a very advanced mechanic. 



# v0.4.0

Fixes and improvements on data_diff.

Improvements:
- data-diff: 
  - Now supports complex data types. Declaring a repeated field (e.g. `"s!.id"` in join_cols will now explode the
    corresponding array and perform the diff on it).
  - When columns are removed or renamed, they are now still displayed in the per-column diff report.
  - Refactored and improved the HTML report: it is now fully standalone and can be opened without any 
    internet connection .
  - Can now generate the HTML report directly on any remote file system accessible by Spark (e.g. "hdfs", "s3", etc.)
  - A user-friendly error is now raised when one of the `join_cols` does not exist. 
- added package `spark_frame.filesystem` that can be used to read and write files directly from the driver using
  the java FileSystem from Spark's JVM.

**Breaking Changes:**
- data-diff:
  - `spark_frame.data_diff.DataframeComparator` object has been removed. 
    Please use directly the method `spark_frame.data_diff.compare_dataframes`.
  - package `spark_frame.data_diff.diff_results` has been renamed to `diff_result`.
  - Generating HTML reports for data diff does not require jinja anymore, but it does now require the installation 
    of the library [data-diff-viewer](https://pypi.org/project/data-diff-viewer/), 
    please check the [Compatibilities and requirements](#compatibilities-and-requirements) 
    section to know which version to use.
  - The DiffResult object returned by the `compare_dataframes` method has evolved. In particular, the
    type of `diff_df_shards` changed from a single `DataFrame` to a `Dict[str, DataFrame]`.
  - `DiffFormatOptions.max_string_length` option has been removed
  - `DiffFormatOptions.nb_diffed_rows` has been renamed to `nb_top_values_kept_per_column`
  - `spark_frame.data_diff.compare_dataframes_impl.DataframeComparatorException` was replaced with
    `spark_frame.exceptions.DataFrameComparisonException`
  - `spark_frame.data_diff.compare_dataframes_impl.CombinatorialExplosionError` was replaced with
    `spark_frame.exceptions.CombinatorialExplosionError`

QA:
- Spark: Added tests to ensure compatibility with Pyspark versions 3.3, 3.4 and 3.5
- Replaced flake and isort with ruff

# v0.3.2

Fixes and improvements on data_diff

- Fix: automatic detection of join_col was sometimes selecting the wrong column
- Visual improvements to HTML diff report:
  - Name of columns used for join are now displayed in bold
  - Total number of column is now displayed when the diff is ok
  - Fix incorrect HTML diff display when one of the DataFrames is empty

# v0.3.1

Fixes and improvements on data_diff

- The `export_html_diff_report` method now accepts arguments to specify the path and encoding of the output html report. 
- Data-diff join now works correctly with null values
- Visual improvements to HTML diff report


# v0.3.0

Fixes and improvements on data_diff

- Fixed incorrect diff results
- Column values are not truncated at all, this was causing incorrect results. The possibility to limit the size 
  of the column values will be added back in a later version
- Made sure that the most frequent values per column are now displayed by decreasing order of frequency


# v0.2.0

Two new exciting features: *analyze* and *data_diff*. 
They are still in experimental stage and will be improved in future releases.

- Added a new transformation `spark_frame.transformations.analyze`.
- Added new *data_diff* feature. Example:

```python
from pyspark.sql import DataFrame
from spark_frame.data_diff import DataframeComparator
df1: DataFrame = ...
df2: DataFrame = ...
diff_result = DataframeComparator().compare_df(df1, df2) # Produces a DiffResult object
diff_result.display() # Print a diff report in the terminal
diff_result.export_to_html() # Generates a html diff report file named diff_report.html
```


# v0.1.1

- Added a new transformation `spark_frame.transformations.flatten_all_arrays`.
- Added support for multi-arg transformation to `nested.select` and `nested.with_fields` 
  With this feature, we can now access parent fields from higher levels
  when applying a transformation. Example:
  
```
>>> nested.print_schema(df)
"""
root
 |-- id: integer (nullable = false)
 |-- s1!.average: integer (nullable = false)
 |-- s1!.values!: integer (nullable = false)
"""
>>> df.show(truncate=False)
+---+--------------------------------------+
|id |s1                                    |
+---+--------------------------------------+
|1  |[{2, [1, 2, 3]}, {3, [1, 2, 3, 4, 5]}]|
+---+--------------------------------------+
>>> new_df = df.transform(nested.with_fields, {
>>>     "s1!.values!": lambda s1, value: value - s1["average"]  # This transformation takes 2 arguments
>>> })
+---+-----------------------------------------+
|id |s1                                       |
+---+-----------------------------------------+
|1  |[{2, [-1, 0, 1]}, {3, [-2, -1, 0, 1, 2]}]|
+---+-----------------------------------------+
```

# v0.1.0

- Added a new _amazing_ module called `spark_frame.nested`, 
  which makes manipulation of nested data structure much easier!
  Make sure to check out the [reference](https://furcypin.github.io/spark-frame/reference/nested/)
  and the [use-cases](https://furcypin.github.io/spark-frame/use_cases/working_with_nested_data/).

- Also added a new module called `spark_frame.nested_functions`,
  which contains aggregation methods for nested data structures
  ([See Reference](https://furcypin.github.io/spark-frame/reference/nested_functions/)).

- New [transformations](https://furcypin.github.io/spark-frame/reference/transformations/):
  - `spark_frame.transformations.transform_all_field_names`
  - `spark_frame.transformations.transform_all_fields`
  - `spark_frame.transformations.unnest_field`
  - `spark_frame.transformations.unnest_all_fields`
  - `spark_frame.transformations.union_dataframes`

# v0.0.3

- New transformation: `spark_frame.transformations.convert_all_maps_to_arrays`.
- New transformation: `spark_frame.transformations.sort_all_arrays`.
- New transformation: `spark_frame.transformations.harmonize_dataframes`.
