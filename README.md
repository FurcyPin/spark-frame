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

[Spark-frame](https://furcypin.github.io/spark-frame/) is a library that brings several utility methods and 
transformation functions for PySpark DataFrames.
These methods were initially part of the [karadoc](https://github.com/FurcyPin/karadoc) project 
used at [Younited](https://medium.com/younited-tech-blog), but they don't rely on karadoc, 
so it makes more sense to keep them as standalone library.

Several of these methods were my initial inspiration to make the cousin project 
[bigquery-frame](https://github.com/FurcyPin/bigquery-frame), which is why you will find similar 
methods in `transformations` and `data_diff` for both `spark_frame` and `bigquery_frame`, except
the former runs on PySpark while the latter runs on BigQuery (obviously).

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
Pyspark must be installed separately to use it.
It is compatible with the following versions:

- Python: requires 3.8.1 or higher (tested against Python 3.9, 3.10 and 3.11)
- pyspark: requires 3.3.0 or higher

This library is tested against Mac and Linux.

# Release notes

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
