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
These methods were initially part of the [karadoc](https://github.com/FurcyPin/karadoc) project 
used at [Younited](https://medium.com/younited-tech-blog), but they were fully independent from karadoc, 
so it made more sense to keep them as a standalone library.

Several of these methods were my initial inspiration to make the cousin project 
[bigquery-frame](https://github.com/FurcyPin/bigquery-frame), which was first made to illustrate
this [blog article](https://medium.com/towards-data-science/sql-jinja-is-not-enough-why-we-need-dataframes-4d71a191936d).
This is why you will find similar methods in both `spark_frame` and `bigquery_frame`, 
except the former runs on PySpark while the latter runs on BigQuery (obviously).
I try to keep both projects consistent together, and new eventually port new developments made on 
one project to the other one.

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

- Python: requires 3.8.1 or higher (tested against Python 3.9, 3.10 and 3.11)
- pyspark: requires 3.3.0 or higher

This library is tested against Windows, Mac and Linux.


**Some features require extra libraries to be installed alongside this project.**
**We chose to not include them as direct dependencies for security and flexibility reasons.**
**This way, users who are not using these features don't need to worry about these dependencies.**

| feature                               | Method                      | module required |
|---------------------------------------|-----------------------------|----------------:|
| Generating HTML reports for data diff | `DiffResult.export_to_html` |          jinja2 |


# Release notes


# v0.2.1

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
