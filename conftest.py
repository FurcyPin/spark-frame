import re
from typing import Union

import pytest


@pytest.fixture()
def fix_pyspark_show_change():
    """Spark changed the way nulls are displayed between versions 3.4 and 3.5
    While nulls used to be written "null", they are now written "NULL".
    This breaks many tests, especially doctests.
    As a workaround, when testing against pyspark versions older than 3.5, we override the DataFrame.show() method
    to replace all occurrences of "null" with "NULL".

    This workaround can be removed when we stop supporting spark versions older than 3.5
    """
    import pyspark

    if pyspark.__version__ < "3.5.0":
        from pyspark.sql import DataFrame
        from spark_frame.utils import show_string

        def show(this: DataFrame, n: int = 20, truncate: Union[bool, int] = True, vertical: bool = False):
            string = show_string(this, n, truncate, vertical)
            pattern = re.compile(r"""(?<!._|: )null""")
            print(re.sub(pattern, "NULL", string))

        DataFrame.show = show
