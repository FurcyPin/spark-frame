from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from spark_frame import functions
from spark_frame.data_diff.package import SAMPLE_ID_COL_NAME

if TYPE_CHECKING:
    from spark_frame.data_diff.diff_result import DiffResult


@lru_cache()
def _get_sample_df_shards_with_cache(diff_result: "DiffResult", max_nb_rows_per_col_state: int) -> List[DataFrame]:
    return _get_sample_df_shards(
        diff_per_col_df=diff_result.get_diff_per_col_df(max_nb_rows_per_col_state),
        diff_df_shards=diff_result.diff_df_shards,
    )


def _get_sample_df_shards(diff_per_col_df: DataFrame, diff_df_shards: Dict[str, DataFrame]) -> List[DataFrame]:
    join_cols_samples_df = (
        diff_per_col_df.select(
            f.explode(
                functions.array_union(
                    f.col("diff.changed.sample_ids"),
                    f.col("diff.no_change.sample_ids"),
                    f.col("diff.only_in_left.sample_ids"),
                    f.col("diff.only_in_right.sample_ids"),
                ),
            ).alias(SAMPLE_ID_COL_NAME),
        )
        .select(f.explode(SAMPLE_ID_COL_NAME).alias(SAMPLE_ID_COL_NAME))
        .distinct()
        .localCheckpoint()
    )
    sample_df_shards = [
        df.withColumn(SAMPLE_ID_COL_NAME, f.col(SAMPLE_ID_COL_NAME)[index]).join(
            join_cols_samples_df,
            on=SAMPLE_ID_COL_NAME,
            how="left_semi",
        )
        for index, df in enumerate(diff_df_shards.values())
    ]
    return sample_df_shards
