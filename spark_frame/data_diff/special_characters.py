from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from spark_frame.conf import REPETITION_MARKER, STRUCT_SEPARATOR
from spark_frame.data_diff.package import REPETITION_MARKER_REPLACEMENT, STRUCT_SEPARATOR_REPLACEMENT

_replacement_mapping = {
    STRUCT_SEPARATOR: STRUCT_SEPARATOR_REPLACEMENT,
    REPETITION_MARKER: REPETITION_MARKER_REPLACEMENT,
}
_replacements = str.maketrans(_replacement_mapping)


def _replace_special_characters(col_name: str) -> str:
    """
    >>> _replace_special_characters("a.b!.c")
    'a__STRUCT__b__ARRAY____STRUCT__c'
    """
    return col_name.translate(_replacements)


def _restore_special_characters(col_name: str) -> str:
    """
    >>> _restore_special_characters("a__STRUCT__b__ARRAY____STRUCT__c")
    'a.b!.c'
    """
    result = col_name
    for value, replacement in _replacement_mapping.items():
        result = result.replace(replacement, value)
    return result


def _replace_special_characters_from_col_names(df: DataFrame) -> DataFrame:
    # TODO: remove this "if" once support for Spark 3.3 is dropped
    if df.sparkSession.version >= "3.4":
        return df.withColumnsRenamed({col: col.translate(_replacements) for col in df.columns})
    else:
        res_df = df
        for col in df.columns:
            res_df = res_df.withColumnRenamed(col, col.translate(_replacements))
        return res_df


def _restore_special_characters_from_col(col: Column) -> Column:
    return f.regexp_replace(
        f.regexp_replace(col, STRUCT_SEPARATOR_REPLACEMENT, STRUCT_SEPARATOR),
        REPETITION_MARKER_REPLACEMENT,
        REPETITION_MARKER,
    )
