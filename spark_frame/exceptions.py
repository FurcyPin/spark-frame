class IllegalArgumentException(Exception):
    """
    Passed an illegal or inappropriate argument.
    """


class SparkSessionNotStarted(Exception):
    """
    When a method using the SparkSession is used but the SparkSession has not been started yet.
    """

    def __init__(self) -> None:
        msg = "The SparkSession must be instantiated before this method can be used"
        Exception.__init__(self, msg)


class FileAlreadyExistsError(Exception):
    """
    When writing a file that already exists.
    """
