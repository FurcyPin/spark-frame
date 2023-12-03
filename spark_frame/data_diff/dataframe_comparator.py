import warnings
from typing import List, Optional

from pyspark.sql import DataFrame

from spark_frame.data_diff import compare_dataframes
from spark_frame.data_diff.diff_results import DiffResult


class DataframeComparator:
    def __init__(self) -> None:
        warning_message = (
            "The class DataframeComparator is deprecated since version 0.3.4. "
            "Please use spark_frame.data_diff.compare_df directly instead."
        )
        warnings.warn(warning_message, category=DeprecationWarning)

    def compare_df(self, left_df: DataFrame, right_df: DataFrame, join_cols: Optional[List[str]] = None) -> DiffResult:
        return compare_dataframes(left_df, right_df, join_cols)
