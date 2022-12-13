# Spark-frame

[![PyPI version](https://badge.fury.io/py/spark-frame.svg)](https://badge.fury.io/py/spark-frame)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spark-frame.svg)](https://pypi.org/project/spark-frame/)
[![GitHub Build](https://img.shields.io/github/workflow/status/FurcyPin/spark-frame/Build%20and%20Validate)](https://github.com/FurcyPin/spark-frame/actions)
[![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=FurcyPin_spark-frame&metric=coverage)](https://sonarcloud.io/component_measures?id=FurcyPin_spark-frame&metric=coverage&view=list)
[![SonarCloud Bugs](https://sonarcloud.io/api/project_badges/measure?project=FurcyPin_spark-frame&metric=bugs)](https://sonarcloud.io/component_measures?metric=reliability_rating&view=list&id=FurcyPin_spark-frame)
[![SonarCloud Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=FurcyPin_spark-frame&metric=vulnerabilities)](https://sonarcloud.io/component_measures?metric=security_rating&view=list&id=FurcyPin_spark-frame)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/spark-frame)](https://pypi.org/project/spark-frame/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## What is it ?

Spark-frame is a library that brings several utility methods and transformation functions for PySpark DataFrames.
These methods were initially part of the [karadoc](https://github.com/FurcyPin/karadoc) project 
used at [Younited](https://medium.com/younited-tech-blog), but they don't rely on karadoc, so it makes more sense 
to keep them as standalone library.

Several of these methods were my initial inspiration to make the cousin project 
[bigquery-frame](https://github.com/FurcyPin/bigquery-frame), which is why you will find similar 
methods in `transformations` and `data_diff` for both `spark_frame` and `bigquery_frame`, except
the former runs on PySpark while the latter runs on BigQuery (obviously).

## Installation

[spark-frame is available on PyPi](https://pypi.org/project/spark-frame/).

```bash
pip install spark-frame
```


# Release notes

# v0.0.3

- New transformation: `spark_frame.transformations.convert_all_maps_to_arrays`.
- New transformation: `spark_frame.transformations.sort_all_arrays`.
- New transformation: `spark_frame.transformations.harmonize_dataframes`.
