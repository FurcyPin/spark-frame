import tempfile
from pathlib import Path
from typing import Optional

import spark_frame
from spark_frame.data_diff.diff_result_summary import DiffResultSummary
from spark_frame.filesystem import write_file
from spark_frame.utils import load_external_module


def export_html_diff_report(
    diff_result_summary: DiffResultSummary,
    title: Optional[str],
    output_file_path: str,
    encoding: str,
) -> None:
    load_external_module("data_diff_viewer", version_constraint="0.2.*")
    from data_diff_viewer import DiffSummary, generate_report_string

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        diff_per_col_parquet_path = temp_dir_path / "diff_per_col"
        diff_result_summary.diff_per_col_df.write.parquet(str(temp_dir_path / "diff_per_col"))
        if title is None:
            report_title = f"{diff_result_summary.left_df_alias} vs {diff_result_summary.right_df_alias}"
        else:
            report_title = title
        column_names_diff = {k: v.value for k, v in diff_result_summary.schema_diff_result.column_names_diff.items()}
        diff_summary = DiffSummary(
            generated_with=f"{spark_frame.__name__}:{spark_frame.__version__}",
            left_df_alias=diff_result_summary.left_df_alias,
            right_df_alias=diff_result_summary.right_df_alias,
            join_cols=diff_result_summary.join_cols,
            same_schema=diff_result_summary.same_schema,
            schema_diff_str=diff_result_summary.schema_diff_result.diff_str,
            column_names_diff=column_names_diff,
            same_data=diff_result_summary.same_data,
            total_nb_rows=diff_result_summary.total_nb_rows,
        )
        report = generate_report_string(
            report_title,
            diff_summary,
            temp_dir_path,
            diff_per_col_parquet_path / "*.parquet",
        )
        write_file(report, output_file_path, mode="overwrite", encoding=encoding)
        print(f"Report exported as {output_file_path}")
