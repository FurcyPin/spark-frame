from dataclasses import dataclass


@dataclass
class DiffFormatOptions:
    """Class used to pass formatting option when displaying a [`DiffResult`][spark_frame.data_diff.DiffResult]"""

    nb_top_values_kept_per_column: int = 10
    """Number of most frequent values/changes kept for each column"""
    left_df_alias: str = "left"
    """Name given to the left DataFrame in the diff"""
    right_df_alias: str = "right"
    """Name given to the right DataFrame in the diff"""
