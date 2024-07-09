import tempfile
from pathlib import Path
from typing import Optional

import spark_frame
from spark_frame import filesystem
from spark_frame.data_diff.diff_result_summary import DiffResultSummary
from spark_frame.filesystem import write_file
from spark_frame.utils import load_external_module


def export_html_diff_report(
    diff_result_summary: DiffResultSummary,
    title: Optional[str],
    output_file_path: str,
    encoding: str,
    base_temp_dir_path: Optional[Path] = None,
) -> None:
    load_external_module("data_diff_viewer", version_constraint="0.3.*")
    from data_diff_viewer import DiffSummary, generate_report_string

    if base_temp_dir_path is None:
        base_temp_dir_path = Path.cwd()
    with tempfile.TemporaryDirectory(dir=base_temp_dir_path) as temp_dir:
        temp_dir_path = Path(temp_dir).absolute()
        remote_diff_per_col_parquet_path = f"{output_file_path}_TEMP_diff_per_col"
        local_diff_per_col_parquet_path = temp_dir_path / "diff_per_col"
        local_diff_per_col_parquet_glob_path = local_diff_per_col_parquet_path / "*.parquet"
        diff_result_summary.diff_per_col_df.write.mode("OVERWRITE").parquet(remote_diff_per_col_parquet_path)
        filesystem.copy_to_local_file(
            remote_diff_per_col_parquet_path,
            str(local_diff_per_col_parquet_path),
            delete_source=True,
        )

        local_sample_parquet_glob_paths = []
        for index, sample_df in enumerate(diff_result_summary.sample_df_shards):
            local_sample_parquet_path = temp_dir_path / f"sample_{index}"
            local_sample_parquet_glob_paths.append(local_sample_parquet_path / "*.parquet")
            remote_sample_parquet_path = f"{output_file_path}_TEMP_sample_{index}"
            sample_df.write.mode("OVERWRITE").parquet(remote_sample_parquet_path)
            filesystem.copy_to_local_file(
                remote_sample_parquet_path,
                str(local_sample_parquet_path),
                delete_source=True,
            )

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
            local_diff_per_col_parquet_glob_path,
            local_sample_parquet_glob_paths,
        )
        write_file(report, output_file_path, mode="overwrite", encoding=encoding)
        print(f"Report exported as {output_file_path}")
