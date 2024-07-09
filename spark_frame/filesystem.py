# mypy: ignore-errors
from typing import Generator

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
    try:
        scala_source_module = getattr(getattr(spark._jvm.scala.io, "Source$"), "MODULE$")  # noqa: SLF001
        scala_source = scala_source_module.fromInputStream(java_file_stream, encoding)
        text = scala_source.mkString()
    finally:
        java_file_stream.close()
    return text


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
    try:
        stream.write(text.encode(encoding))
        stream.flush()
    finally:
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


def copy_to_local_file(source_path: str, dest_path: str, delete_source: bool) -> None:
    r"""Copy a file from a remote storage to a local storage using `org.apache.hadoop.fs.FileSystem` from Spark's JVM.

    Depending on how Spark is configured, it can read from any file system supported by Spark.
    (like "file://", "hdfs://", "s3://", "gs://", "abfs://", etc.)

    !!! warning
        The SparkSession must be instantiated before this method can be used.

    Args:
        source_path: Path of the remote source file
        dest_path: Path of thelocal destination file
        delete_source: Delete the source if True

    Raises:
        spark_frame.exceptions.SparkSessionNotStarted: if the SparkSession is not started when this method is called.

    Examples: Example 1: copying a file without deleting the source
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> write_file(text="Hello\n", path="test_working_dir/write_file_1.txt", mode="overwrite")
        >>> copy_to_local_file(
        ...     "test_working_dir/write_file_1.txt",
        ...     "test_working_dir/write_file_2.txt",
        ...     delete_source=False
        ...     )
        >>> text = read_file("test_working_dir/write_file_2.txt")
        >>> print(text)
        Hello
        <BLANKLINE>
        >>> text = read_file("test_working_dir/write_file_1.txt")
        >>> print(text)
        Hello
        <BLANKLINE>

    Examples: Example 2: copying a file to a folder that does not exists
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> write_file(text="Hello\n", path="test_working_dir/write_file_1.txt", mode="overwrite")
        >>> copy_to_local_file(
        ...     "test_working_dir/write_file_1.txt",
        ...     "test_working_dir/sub_dir/write_file_2.txt",
        ...     delete_source=False
        ...     )
        >>> text = read_file("test_working_dir/sub_dir/write_file_2.txt")
        >>> print(text)
        Hello
        <BLANKLINE>

    Examples: Example 2: copying a file and deleting the source file
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> write_file(text="Hello\n", path="test_working_dir/write_file_1.txt", mode="overwrite")
        >>> copy_to_local_file(
        ...     "test_working_dir/write_file_1.txt",
        ...     "test_working_dir/write_file_3.txt",
        ...     delete_source=True
        ...     )
        >>> text = read_file("test_working_dir/write_file_3.txt")
        >>> print(text)
        Hello
        <BLANKLINE>
        >>> text = read_file("test_working_dir/write_file_1.txt")
        Traceback (most recent call last):
          ...
        FileNotFoundError: The file test_working_dir/write_file_1.txt was not found.
    """
    spark = SparkSession.getActiveSession()
    assert_true(spark is not None, SparkSessionNotStarted())
    java_fs = _get_java_file_system(source_path, spark)

    java_source_path = spark._jvm.org.apache.hadoop.fs.Path(source_path)  # noqa: SLF001
    java_dest_path = spark._jvm.org.apache.hadoop.fs.Path(dest_path)  # noqa: SLF001

    java_fs.copyToLocalFile(delete_source, java_source_path, java_dest_path)


def copy_to_local_folder(source_folder: str, dest_folder: str, delete_source: bool) -> None:
    r"""Recursively copy all files and sub-folders in the source folder from a remote storage to a local storage
    using `org.apache.hadoop.fs.FileSystem` from Spark's JVM.

    Depending on how Spark is configured, it can read from any file system supported by Spark.
    (like "file://", "hdfs://", "s3://", "gs://", "abfs://", etc.)

    !!! warning
        The SparkSession must be instantiated before this method can be used.

    Args:
        source_folder: Path of the remote source folder
        dest_folder: Path of the local destination folder
        delete_source: Delete the source if True

    Raises:
        spark_frame.exceptions.SparkSessionNotStarted: if the SparkSession is not started when this method is called.

    Examples: Example 1: copying a folder without deleting the source
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> write_file(text="Hello\n", path="test_working_dir/source_folder/sub_folder/write_file_1.txt", mode="overwrite")
        >>> write_file(text="Hello\n", path="test_working_dir/source_folder/sub_folder/write_file_2.txt", mode="overwrite")
        >>> copy_to_local_folder(
        ...     "test_working_dir/source_folder",
        ...     "test_working_dir/dest_folder",
        ...     delete_source=False
        ...     )
        >>> text = read_file("test_working_dir/dest_folder/sub_folder/write_file_1.txt")
        >>> print(text)
        Hello
        <BLANKLINE>
        >>> text = read_file("test_working_dir/dest_folder/sub_folder/write_file_2.txt")
        >>> print(text)
        Hello
        <BLANKLINE>
        >>> text = read_file("test_working_dir/source_folder/sub_folder/write_file_1.txt")
        >>> print(text)
        Hello
        <BLANKLINE>
        >>> text = read_file("test_working_dir/source_folder/sub_folder/write_file_2.txt")
        >>> print(text)
        Hello
        <BLANKLINE>

    Examples: Example 1: copying a folder and deleting the source
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> write_file(text="Hello\n", path="test_working_dir/source_folder/sub_folder/write_file_1.txt", mode="overwrite")
        >>> write_file(text="Hello\n", path="test_working_dir/source_folder/sub_folder/write_file_2.txt", mode="overwrite")
        >>> copy_to_local_folder(
        ...     "test_working_dir/source_folder",
        ...     "test_working_dir/dest_folder",
        ...     delete_source=True
        ...     )
        >>> text = read_file("test_working_dir/dest_folder/sub_folder/write_file_1.txt")
        >>> print(text)
        Hello
        <BLANKLINE>
        >>> text = read_file("test_working_dir/dest_folder/sub_folder/write_file_2.txt")
        >>> print(text)
        Hello
        <BLANKLINE>
        >>> text = read_file("test_working_dir/source_folder/sub_folder/write_file_1.txt")
        Traceback (most recent call last):
          ...
        FileNotFoundError: The file test_working_dir/source_folder/sub_folder/write_file_1.txt was not found.
        >>> text = read_file("test_working_dir/source_folder/sub_folder/write_file_2.txt")
        Traceback (most recent call last):
          ...
        FileNotFoundError: The file test_working_dir/source_folder/sub_folder/write_file_2.txt was not found.
    """  # noqa: E501
    spark = SparkSession.getActiveSession()
    assert_true(spark is not None, SparkSessionNotStarted())

    java_fs = _get_java_file_system(source_folder, spark)
    java_source_folder = spark._jvm.org.apache.hadoop.fs.Path(source_folder)  # noqa: SLF001
    source_folder = java_fs.getFileStatus(java_source_folder).getPath().toUri().toString()
    for source_file in recursively_list_all_files_in_folder(source_folder):
        relative_source_file = _remove_prefix(source_file, source_folder)
        dest_file = dest_folder + relative_source_file
        copy_to_local_file(source_file, dest_file, delete_source=delete_source)
    if delete_source:
        java_source_folder = spark._jvm.org.apache.hadoop.fs.Path(source_folder)  # noqa: SLF001
        java_fs = _get_java_file_system(source_folder, spark)
        java_fs.delete(java_source_folder, True)  # noqa: FBT003


# TODO: once Python 3.8 is deprecated, we can replace this with String.remove_prefix()
def _remove_prefix(string: str, prefix: str) -> str:
    """Remove a prefix from a string.

    >>> _remove_prefix("abc", "a")
    'bc'
    >>> _remove_prefix("abc", "b")
    Traceback (most recent call last):
        ...
    spark_frame.exceptions.IllegalArgumentException: b is not a prefix of abc

    """
    if string.startswith(prefix):
        return string[len(prefix) :]
    else:
        message = f"{prefix} is not a prefix of {string}"
        raise IllegalArgumentException(message)


def recursively_list_all_files_in_folder(folder_path: str) -> Generator[str, None, None]:
    r"""Recursively list all files in the given folder and its sub-folders on a remote storage
    using `org.apache.hadoop.fs.FileSystem` from Spark's JVM.

    Depending on how Spark is configured, it can read from any file system supported by Spark.
    (like "file://", "hdfs://", "s3://", "gs://", "abfs://", etc.)

    !!! warning
        The SparkSession must be instantiated before this method can be used.

    Args:
        folder_path: Path of the remote folder

    Raises:
        spark_frame.exceptions.SparkSessionNotStarted: if the SparkSession is not started when this method is called.

    Examples: Example 1: Listing all files in a folder
        >>> spark = SparkSession.builder.appName("doctest").getOrCreate()
        >>> write_file(text="Hello\n", path="test_working_dir/source_folder/sub_folder/write_file_1.txt", mode="overwrite")
        >>> write_file(text="Hello\n", path="test_working_dir/source_folder/sub_folder/write_file_2.txt", mode="overwrite")
        >>> for file in sorted(recursively_list_all_files_in_folder("test_working_dir/source_folder")):
        ...     print("File:" + file)  # doctest: +ELLIPSIS
        File:.../test_working_dir/source_folder/sub_folder/write_file_1.txt
        File:.../test_working_dir/source_folder/sub_folder/write_file_2.txt

    Examples: Example 2: When listing a file, it should still work even if it is not a folder
        >>> for file in recursively_list_all_files_in_folder(
        ...     "test_working_dir/source_folder/sub_folder/write_file_1.txt"
        ... ):
        ...     print("File:" + file)  # doctest: +ELLIPSIS
        File:.../test_working_dir/source_folder/sub_folder/write_file_1.txt
    """  # noqa: E501
    spark = SparkSession.getActiveSession()
    assert_true(spark is not None, SparkSessionNotStarted())
    java_fs = _get_java_file_system(folder_path, spark)
    java_folder_path = spark._jvm.org.apache.hadoop.fs.Path(folder_path)  # noqa: SLF001
    for file_status in java_fs.listStatus(java_folder_path):
        if file_status.isDirectory():
            yield from recursively_list_all_files_in_folder(file_status.getPath().toUri().toString())
        if file_status.isFile():
            yield file_status.getPath().toUri().toString()
