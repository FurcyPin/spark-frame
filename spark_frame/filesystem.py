# mypy: ignore-errors
from py4j import java_gateway
from py4j.java_gateway import JavaObject
from pyspark.sql import SparkSession

from spark_frame.exceptions import FileAlreadyExistsError, IllegalArgumentException, SparkSessionNotStarted
from spark_frame.utils import assert_true

MODE_OVERWRITE = "overwrite"
MODE_APPEND = "append"
MODE_ERROR_IF_EXISTS = "error_if_exists"

VALID_MODES = [MODE_OVERWRITE, MODE_APPEND, MODE_ERROR_IF_EXISTS]


def read_file(path: str, encoding: str = "utf8") -> str:
    r"""Read the content of a file using the `org.apache.hadoop.fs.FileSystem` from Spark's JVM.
    Depending on how Spark is configured, it can write on any file system supported by Spark.
    (like "file://", "hdfs://", "s3://", "gs://", "abfs://", etc.)

    !!! warning
        This method loads the entirety of the file in memory as a Python str object.
        It's use should be restricted to reading small files such as configuration files or reports.

        When reading large data files, it is recommended to use the `spark.read` method.

    !!! warning
        The SparkSession must be instantiated before this method can be used.

    Args:
        path: The path of the file to read
        encoding:

    Raises:
        spark_frame.exceptions.SparkSessionNotStarted: if the SparkSession is not started when this method is called.
        FileNotFoundError: if no file were found at the specified path.

    Returns:
        the content of the file.

    Examples:
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> write_file(text="Hello\n", path="test_working_dir/read_file.txt", mode="overwrite")
        >>> text = read_file("test_working_dir/read_file.txt")
        >>> print(text)
        Hello
        <BLANKLINE>
        >>> read_file("test_working_dir/this_file_does_not_exist.txt")
        Traceback (most recent call last):
          ...
        FileNotFoundError: The file test_working_dir/this_file_does_not_exist.txt was not found.
    """
    spark = SparkSession.getActiveSession()
    assert_true(spark is not None, SparkSessionNotStarted())
    java_fs = _get_java_file_system(path, spark)
    java_path = spark._jvm.org.apache.hadoop.fs.Path(path)  # noqa: SLF001
    if not java_fs.exists(java_path):
        msg = f"The file {path} was not found."
        raise FileNotFoundError(msg)
    java_file_stream = java_fs.open(java_path)
    scala_source_module = getattr(getattr(spark._jvm.scala.io, "Source$"), "MODULE$")  # noqa: SLF001
    scala_source = scala_source_module.fromInputStream(java_file_stream, encoding)
    return scala_source.mkString()


def write_file(
    text: str,
    path: str,
    mode: str = MODE_ERROR_IF_EXISTS,
    encoding: str = "utf8",
) -> None:
    r"""Write given text to a file using the `org.apache.hadoop.fs.FileSystem` from Spark's JVM.
    Depending on how Spark is configured, it can write on any file system supported by Spark.
    (like "file://", "hdfs://", "s3://", "gs://", "abfs://", etc.)

    !!! warning
        This method loads the entirety of the file in memory as a Python str object.
        It's use should be restricted to writing small files such as configuration files or reports.

        When reading large data files, it is recommended to use the `spark.read` method.

    !!! warning
        The SparkSession must be instantiated before this method can be used.

    Args:
        text: The content of the file to write
        path: The path of the file to write
        mode: Either one of ["overwrite", "append", "error_if_exists"].
        encoding: Encoding to use

    Raises:
        spark_frame.exceptions.SparkSessionNotStarted: if the SparkSession is not started when this method is called.
        spark_frame.exceptions.IllegalArgumentException: if the mode is incorrect.
        spark_frame.exceptions.FileAlreadyExistsError: if the file already exists and mode = "error_if_exists".

    Examples:
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> write_file(text="Hello\n", path="test_working_dir/write_file.txt", mode="overwrite")
        >>> text = read_file("test_working_dir/write_file.txt")
        >>> print(text)
        Hello
        <BLANKLINE>
        >>> write_file(text="World\n", path="test_working_dir/write_file.txt", mode="append")
        >>> text = read_file("test_working_dir/write_file.txt")
        >>> print(text)
        Hello
        World
        <BLANKLINE>
        >>> write_file(text="Never mind\n", path="test_working_dir/write_file.txt", mode="overwrite")
        >>> text = read_file("test_working_dir/write_file.txt")
        >>> print(text)
        Never mind
        <BLANKLINE>
        >>> write_file(text="Never mind\n", path="test_working_dir/write_file.txt", mode="error_if_exists")
        Traceback (most recent call last):
            ...
        spark_frame.exceptions.FileAlreadyExistsError: The file test_working_dir/write_file.txt already exists.
        >>> write_file(text="Never mind\n", path="test_working_dir/write_file.txt", mode="incorrect_mode")
        Traceback (most recent call last):
            ...
        spark_frame.exceptions.IllegalArgumentException: Invalid write mode: incorrect_mode. Accepted modes are: ['overwrite', 'append', 'error_if_exists']
    """  # noqa: E501
    spark = SparkSession.getActiveSession()
    assert_true(spark is not None, SparkSessionNotStarted())

    java_fs = _get_java_file_system(path, spark)
    java_path = spark._jvm.org.apache.hadoop.fs.Path(path)  # noqa: SLF001

    if mode.lower() not in VALID_MODES:
        msg = f"Invalid write mode: {mode}. Accepted modes are: {VALID_MODES}"
        raise IllegalArgumentException(msg)
    if not java_fs.exists(java_path):
        stream = java_fs.create(java_path)
    else:
        if mode == MODE_ERROR_IF_EXISTS:
            msg = f"The file {path} already exists."
            raise FileAlreadyExistsError(msg)
        if mode == MODE_OVERWRITE:
            stream = java_fs.create(java_path)
        elif mode == MODE_APPEND:
            stream = java_fs.append(java_path)
    stream.write(text.encode(encoding))
    stream.flush()
    stream.close()


def _get_java_file_system(path: str, spark: SparkSession) -> JavaObject:
    java_uri = spark._jvm.java.net.URI(path)  # noqa: SLF001
    java_filesystem = spark._jvm.org.apache.hadoop.fs.FileSystem  # noqa: SLF001
    java_fs = java_filesystem.get(java_uri, spark._jsc.hadoopConfiguration())  # noqa: SLF001
    checksum_file_system = spark._jvm.org.apache.hadoop.fs.ChecksumFileSystem  # noqa: SLF001
    # When Spark is running in local mode, the FileSystem class used is `ChecksumFileSystem`, which does not implement
    # the `append` method. Oddly enough, it wraps a `RawLocalFileSystem` object which does implement `append`,
    # so we return this `RawLocalFileSystem` object instead.
    if java_gateway.is_instance_of(spark.sparkContext._gateway, java_fs, checksum_file_system):  # noqa: SLF001
        java_fs = java_fs.getRawFileSystem()
    return java_fs
