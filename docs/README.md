{%
   include-markdown "../README.md"
%}

## Usage

Spark-frame contains several utility methods, all documented in the [reference](/spark-frame/reference/functions). 
There are grouped into several modules:

- [functions](reference#spark_framefunctions): 
  Extra functions similar to [pyspark.sql.functions](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html).
- [graph](reference#spark_framegraph):
  Implementations of graph algorithms.
- [transformations](reference#spark_frametransformations):
  Generic transformations taking one or more input DataFrames as argument and returning a new DataFrame.
- [schema_utils](reference#spark_frameschema_utils):
  Methods useful for manipulating DataFrame schemas.


