from typing import Optional

import jinja2
import pkg_resources

from spark_frame.data_diff.diff_result_summary import DiffResultSummary


def export_html_diff_report(diff_result_summary: DiffResultSummary, title: Optional[str] = None) -> None:
    if title is None:
        title_str = f"{diff_result_summary.left_df_alias} vs {diff_result_summary.right_df_alias}"
    else:
        title_str = title
    # Load the Jinja2 template
    template_str = pkg_resources.resource_string("spark_frame", "templates/diff_report.html.jinja2")
    template = jinja2.Template(template_str.decode("utf-8"))

    # Render the template with the DiffResultSummary object
    html = template.render(
        title=title_str,
        diff_result_summary=diff_result_summary,
        diff_per_col=diff_result_summary.diff_per_col_df.collect(),
    )

    # Save the rendered HTML to a file
    with open("diff_report.html", "w") as f:
        f.write(html)

    print("Report exported as diff_report.html")
