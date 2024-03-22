import typing
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import StructType

from spark_frame import functions
from spark_frame.data_diff.package import EXISTS_COL_NAME, IS_EQUAL_COL_NAME, SAMPLE_ID_COL_NAME, canonize_col
from spark_frame.utils import quote

if TYPE_CHECKING:
    from spark_frame.data_diff.diff_result import DiffResult


@lru_cache()
def _get_sample_df_shards_with_cache(diff_result: "DiffResult", max_nb_rows_per_col_state: int) -> List[DataFrame]:
    return _get_sample_df_shards(
        diff_per_col_df=diff_result.get_diff_per_col_df(max_nb_rows_per_col_state),
        diff_df_shards=diff_result.diff_df_shards,
    )


def _get_sample_df_shard(df: DataFrame, join_cols_samples_df: DataFrame, index: int) -> DataFrame:
    df = df.withColumn(SAMPLE_ID_COL_NAME, f.col(SAMPLE_ID_COL_NAME)[index]).join(
        join_cols_samples_df,
        on=SAMPLE_ID_COL_NAME,
        how="left_semi",
    )
    df = df.select(
        SAMPLE_ID_COL_NAME,
        *[
            f.struct(
                canonize_col(
                    df[quote(field.name) + ".left_value"],
                    typing.cast(StructType, field.dataType).fields[0],
                )
                .cast("STRING")
                .alias("left_value"),
                canonize_col(
                    df[quote(field.name) + ".right_value"],
                    typing.cast(StructType, field.dataType).fields[1],
                )
                .cast("STRING")
                .alias("right_value"),
                df[quote(field.name) + ".exists_left"].alias("exists_left"),
                df[quote(field.name) + ".exists_right"].alias("exists_right"),
            ).alias(field.name)
            for field in df.schema.fields
            if field.name not in [SAMPLE_ID_COL_NAME, EXISTS_COL_NAME, IS_EQUAL_COL_NAME]
        ],
    )
    return df


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
        _get_sample_df_shard(df, join_cols_samples_df, index) for index, df in enumerate(diff_df_shards.values())
    ]
    return sample_df_shards
