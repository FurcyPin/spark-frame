from dataclasses import dataclass

DEFAULT_NB_DIFFED_ROWS = 10
DEFAULT_LEFT_DF_ALIAS = "left"
DEFAULT_RIGHT_DF_ALIAS = "right"


@dataclass
class DiffFormatOptions:
    nb_top_values_kept_per_column: int = DEFAULT_NB_DIFFED_ROWS
    """Number of most frequent values/changes kept for each column"""
    left_df_alias: str = DEFAULT_LEFT_DF_ALIAS
    """Name given to the left DataFrame in the diff"""
    right_df_alias: str = DEFAULT_RIGHT_DF_ALIAS
    """Name given to the right DataFrame in the diff"""
