from pyspark.sql import SparkSession


def _get_sample_data():
    spark = SparkSession.builder.appName("doctest").getOrCreate()
    df = spark.createDataFrame(
        [
            (
                1,
                '{"model_name": "bot_detector", "model_version": 3, "model_args": "some data"}',
                '{"model_score": 0.94654, "model_parameters": "some data"}',
            ),
            (
                2,
                '{"model_name": "cat_finder", "model_version": 3, "model_args": "some data"}',
                '{"model_score": 0.4234, "model_parameters": "some data"}',
            ),
        ],
        "call_id INT, raw_input STRING, raw_output STRING",
    ).repartition(1)
    return df


def extracting_json_values():
    """Sometimes, a column in a data source contains raw json strings, and you want to extract this value before
    starting to understand it.

    This already happened to me in several cases, such as:
    - Some automatic data capture tool that wraps a payload's raw json value into an Avro file.
    - A microservice event that follows a data contract but one of the column contains the raw json payload that this
      microservice exchanged with another external API.

    Examples: Let's take a sample DataFrame with two raw json columns.

        >>> from spark_frame.examples.working_with_json import _get_sample_data
        >>> df = _get_sample_data()
        >>> df.printSchema()
        root
         |-- call_id: integer (nullable = true)
         |-- raw_input: string (nullable = true)
         |-- raw_output: string (nullable = true)
        <BLANKLINE>
        >>> df.show(truncate=False)  # noqa: E501 # doctest: +NORMALIZE_WHITESPACE
        +-------+-----------------------------------------------------------------------------+---------------------------------------------------------+
        |call_id|raw_input                                                                    |raw_output                                               |
        +-------+-----------------------------------------------------------------------------+---------------------------------------------------------+
        |1      |{"model_name": "bot_detector", "model_version": 3, "model_args": "some data"}|{"model_score": 0.94654, "model_parameters": "some data"}|
        |2      |{"model_name": "cat_finder", "model_version": 3, "model_args": "some data"}  |{"model_score": 0.4234, "model_parameters": "some data"} |
        +-------+-----------------------------------------------------------------------------+---------------------------------------------------------+
        <BLANKLINE>

        This DataFrame represents the logs of an application calling a machine learning model. Keeping the "call_id" is
        important to be able to link this call to other events that happen in the system, and we would like to analyze
        these logs with typed data.

        ### Without spark-frame

        Spark does provide a [`from_json`](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.from_json.html)
        function that can parse a raw json column and convert it into a struct, but it does require the user to provide
        the schema of the json column in advance, like this:

        >>> from pyspark.sql import functions as f
        >>> raw_input_schema = '{"fields":[{"name":"model_name","nullable":true,"type":"string"},{"name":"model_version","nullable":true,"type":"integer"},{"name":"model_args","nullable":true,"type":"string"}],"type":"struct"}'
        >>> raw_output_schema = '{"fields":[{"name":"model_score","nullable":true,"type":"double"},{"name":"model_parameters","nullable":true,"type":"string"}],"type":"struct"}'
        >>> df.withColumn(
        ...     "raw_input", f.from_json("raw_input", raw_input_schema)
        ... ).withColumn(
        ...     "raw_output", f.from_json("raw_output", raw_output_schema)
        ... ).show(truncate=False)
        +-------+----------------------------+--------------------+
        |call_id|raw_input                   |raw_output          |
        +-------+----------------------------+--------------------+
        |1      |{bot_detector, 3, some data}|{0.94654, some data}|
        |2      |{cat_finder, 3, some data}  |{0.4234, some data} |
        +-------+----------------------------+--------------------+
        <BLANKLINE>

        ### With spark-frame

        While it does works, as you can see writing the schema can be quite heavy.
        Also, for some reason, `from_json` does not accept the "simpleString" format, unlike
        the [`SparkSession.createDataFrame`](
        https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.SparkSession.createDataFrame.html)
        method.
        The first thing we can do to make things simpler is by using the method
        [`spark_frame.schema_utils.schema_from_simple_string`](
        /reference/#spark_frame.schema_utils.schema_from_simple_string) like this :

        >>> from spark_frame.schema_utils import schema_from_simple_string
        >>> raw_input_schema = schema_from_simple_string("model_name: STRING, model_version: INT, model_args: STRING")
        >>> raw_output_schema = schema_from_simple_string("model_score: DOUBLE, model_parameters: STRING")
        >>> df.withColumn(
        ...     "raw_input", f.from_json("raw_input", raw_input_schema)
        ... ).withColumn(
        ...     "raw_output", f.from_json("raw_output", raw_output_schema)
        ... ).show(truncate=False)
        +-------+----------------------------+--------------------+
        |call_id|raw_input                   |raw_output          |
        +-------+----------------------------+--------------------+
        |1      |{bot_detector, 3, some data}|{0.94654, some data}|
        |2      |{cat_finder, 3, some data}  |{0.4234, some data} |
        +-------+----------------------------+--------------------+
        <BLANKLINE>

        But if we don't know the schema or if we know that the schema may evolve and we want to add
        (or at least, detect) the new fields automatically, we can leverage Spark's automatic json schema inference
        by using the method [spark_frame.transformations.parse_json_columns](
        /reference/#spark_frame.transformations_impl.parse_json_columns.parse_json_columns) to infer automatically
        the schema of these json columns.

        >>> from spark_frame.transformations import parse_json_columns
        >>> res = parse_json_columns(df, ["raw_input", "raw_output"])
        >>> res.show(truncate=False)
        +-------+----------------------------+--------------------+
        |call_id|raw_input                   |raw_output          |
        +-------+----------------------------+--------------------+
        |1      |{some data, bot_detector, 3}|{some data, 0.94654}|
        |2      |{some data, cat_finder, 3}  |{some data, 0.4234} |
        +-------+----------------------------+--------------------+
        <BLANKLINE>
        >>> res.printSchema()
        root
         |-- call_id: integer (nullable = true)
         |-- raw_input: struct (nullable = true)
         |    |-- model_args: string (nullable = true)
         |    |-- model_name: string (nullable = true)
         |    |-- model_version: long (nullable = true)
         |-- raw_output: struct (nullable = true)
         |    |-- model_parameters: string (nullable = true)
         |    |-- model_score: double (nullable = true)
        <BLANKLINE>

        As we can see, the order of the field is different, this is because Spark's automatic inference will always
        sort the json field by names.
    """
    # This is a hacky way to have doctests that runs in the pipeline and are usable in the doc thanks to mkdocstrings
