#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
report_html.py

SpaPhish report generator (HTML + optional PDF via WeasyPrint).

Features:
- Zero hardcoded section / figure / table numbers — everything is
  assigned automatically at render time.
- Automatic numbering:
    Sections:    1 … N
    Subsections: sec.sub  (within each section)
    Figures:     Figure 1 … F  (global)
    Tables:      Table  1 … T  (global)
- Stable DOM IDs for navigation anchors.
- "Back to menu" link at the end of every section.
- Print-optimised CSS so that tables wrap and never overflow the page width.

Prerequisites:
  Run analysis/analyze_dataset.py first to populate:
    <project_root>/output/tables/*.csv
    <project_root>/output/figures/*.png

Outputs:
  <project_root>/output/index_updated.html
  <project_root>/output/SpaPhish_Report_UPDATED.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import html
import json
import re
import sys

import pandas as pd


# =============================================================================
# Helpers (I/O + HTML)
# =============================================================================

def load_dataset_schema_json(tables_dir: Path):
    """
    Tries to load dataset_schema_json.json from:
      - output/tables/dataset_schema_json.json
      - output/dataset_schema_json.json
    Returns JSON object (dict/list) or None.
    """
    candidates = [
        tables_dir / "dataset_schema_json.json",
        tables_dir.parent / "dataset_schema_json.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[WARN] Could not load schema JSON from {path}: {exc}")
                return None
    print("[WARN] dataset_schema_json.json not found.")
    return None


def load_table(tables_dir: Path, filename: str) -> pd.DataFrame | None:
    """Load a CSV table from the report tables directory and normalize its columns."""
    path = tables_dir / filename
    if not path.exists():
        print(f"[WARN] Table not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        # Drop Unnamed columns
        df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]

        # Normalize + de-duplicate column names
        normalized = []
        for c in df.columns:
            name = str(c).strip() or "column"
            normalized.append(name)

        seen: Dict[str, int] = {}
        unique_cols = []
        for name in normalized:
            if name not in seen:
                seen[name] = 0
                unique_cols.append(name)
            else:
                seen[name] += 1
                unique_cols.append(f"{name}_{seen[name]}")
        df.columns = unique_cols
        return df
    except Exception as e:
        print(f"[WARN] Failed to read table {path}: {e}")
        return None


def img_tag(figs_dir: Path, filename: str, alt: str) -> str:
    """Return an HTML img tag for a generated figure."""
    path = figs_dir / filename
    if not path.exists():
        print(f"[WARN] Figure not found: {path}")
        return ""
    src = f"figures/{filename}"
    return f'<img src="{html.escape(src)}" alt="{html.escape(alt)}" class="figure-img"/>'


def table_preview_html(df, max_rows: int = 10, max_cols: int = 10) -> str:
    """
    Render a small HTML preview for a pandas DataFrame.
    - Limits rows/cols.
    - Adds an index column.
    """
    if df is None:
        return "<p class='warning'>Table not available.</p>"

    if not isinstance(df, pd.DataFrame) or df.empty:
        return "<p class='warning'>Table is empty or invalid.</p>"

    df_preview = df.iloc[:max_rows, :max_cols]

    header_cells = ["<th class='table-index'>Row</th>"]
    for col in df_preview.columns:
        header_cells.append(f"<th>{html.escape(str(col))}</th>")
    header_html = "<tr>" + "".join(header_cells) + "</tr>"

    body_rows = []
    for idx, row in df_preview.iterrows():
        row_cells = [f"<td class='table-index'>{html.escape(str(idx))}</td>"]
        for val in row.values:
            row_cells.append(f"<td>{html.escape(str(val))}</td>")
        body_rows.append("<tr>" + "".join(row_cells) + "</tr>")

    body_html = "\n".join(body_rows)

    return f"""
    <div class="table-wrapper">
      <table class="data-table">
        <thead>{header_html}</thead>
        <tbody>{body_html}</tbody>
      </table>
    </div>
    """


def download_link(filename: str, label: str) -> str:
    """Return an HTML download link for a report artifact."""
    href = f"tables/{filename}"
    return f'<a class="download-link" href="{html.escape(href)}" download>{html.escape(label)}</a>'


def _slug(s: str) -> str:
    """Return a slugified token suitable for IDs and filenames."""
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-_]+", "", s)
    s = s.strip("-_")
    return s or "na"


# =============================================================================
# Report Engine (NO hardcoded numbering)
# =============================================================================

@dataclass
class Block:
    kind: str                  # "p" | "raw" | "h3" | "figure" | "table"
    title: Optional[str] = None
    html: Optional[str] = None
    uid: Optional[str] = None  # stable id for figure/table


@dataclass
class Section:
    sec_id: str
    title: str
    blocks: List[Block] = field(default_factory=list)

    def p(self, text_html: str) -> "Section":
        """Perform the P operation."""
        self.blocks.append(Block(kind="p", html=text_html))
        return self

    def raw(self, raw_html: str) -> "Section":
        """Perform the Raw operation."""
        self.blocks.append(Block(kind="raw", html=raw_html))
        return self

    def subsection(self, title: str) -> "Section":
        """Perform the Subsection operation."""
        self.blocks.append(Block(kind="h3", title=title))
        return self

    def figure(self, fig_id: str, caption: str, inner_html: str) -> "Section":
        """Perform the Figure operation."""
        self.blocks.append(Block(kind="figure", uid=fig_id, title=caption, html=inner_html))
        return self

    def table(self, tbl_id: str, caption: str, inner_html: str) -> "Section":
        """Perform the Table operation."""
        self.blocks.append(Block(kind="table", uid=tbl_id, title=caption, html=inner_html))
        return self


@dataclass
class Report:
    title: str
    subtitle: str
    sections: List[Section] = field(default_factory=list)

    def add_section(self, sec_id: str, title: str) -> Section:
        """Perform the Add section operation."""
        s = Section(sec_id=sec_id, title=title)
        self.sections.append(s)
        return s

    def render_html(self) -> str:
        """Perform the Render HTML operation."""
        fig_counter = 0
        tbl_counter = 0

        # NAV
        nav_links = []
        for i, sec in enumerate(self.sections, start=1):
            nav_links.append(f'<a href="#{html.escape(sec.sec_id)}">{i}. {html.escape(sec.title)}</a>')
        nav_html = (
            '<nav id="menu">'
            '<strong>Sections:</strong> '
            + "\n        ".join(nav_links) +
            '</nav>'
        )

        # BODY
        body_parts: List[str] = []
        for sec_idx, sec in enumerate(self.sections, start=1):
            sub_idx = 0
            parts = [f'<section id="{html.escape(sec.sec_id)}">']
            parts.append(f"<h2>{sec_idx}. {html.escape(sec.title)}</h2>")

            for b in sec.blocks:
                if b.kind == "p":
                    parts.append(f"<p>{b.html or ''}</p>")
                elif b.kind == "raw":
                    parts.append(b.html or "")
                elif b.kind == "h3":
                    sub_idx += 1
                    parts.append(f"<h3>{sec_idx}.{sub_idx} {html.escape(b.title or '')}</h3>")
                elif b.kind == "figure":
                    if not b.uid:
                        raise ValueError("Figure block without fig_id (uid).")
                    fig_counter += 1
                    fig_dom_id = f"fig-{_slug(b.uid)}"
                    parts.append(f'<div class="figure-block" id="{html.escape(fig_dom_id)}">')
                    parts.append(f"<h3>Figure {fig_counter}. {html.escape(b.title or '')}</h3>")
                    parts.append(b.html or "")
                    parts.append("</div>")
                elif b.kind == "table":
                    if not b.uid:
                        raise ValueError("Table block without tbl_id (uid).")
                    tbl_counter += 1
                    tbl_dom_id = f"tbl-{_slug(b.uid)}"
                    parts.append(f'<div class="table-block" id="{html.escape(tbl_dom_id)}">')
                    parts.append(f"<h3>Table {tbl_counter}. {html.escape(b.title or '')}</h3>")
                    parts.append(b.html or "")
                    parts.append("</div>")
                else:
                    raise ValueError(f"Unknown block kind: {b.kind}")

            parts.append('<p class="back-to-menu"><a href="#menu">↑ Volver al menú</a></p>')
            parts.append("</section>")
            body_parts.append("\n".join(parts))

        # CSS (includes print rules for PDF)
        css = r"""
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f6fa;
    color: #222;
    line-height: 1.5;
  }
  header {
    background: #243447;
    color: #fff;
    padding: 20px 40px;
  }
  header h1 {
    margin: 0 0 8px;
    font-size: 24px;
  }
  header p {
    margin: 0;
    font-size: 14px;
    opacity: 0.9;
  }
  .container {
    max-width: 1100px;
    margin: 20px auto 40px;
    padding: 0 16px;
  }
  nav {
    background: #ffffff;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    font-size: 14px;
  }
  nav a {
    margin-right: 12px;
    text-decoration: none;
    color: #1a73e8;
  }
  nav a:hover {
    text-decoration: underline;
  }
  section {
    background: #ffffff;
    border-radius: 8px;
    padding: 18px 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  section h2 {
    margin-top: 0;
    border-bottom: 1px solid #e2e4ea;
    padding-bottom: 6px;
  }
  .figure-block {
    margin: 18px 0;
    text-align: center;
  }
  .figure-block h3 {
    margin-bottom: 8px;
    font-size: 15px;
  }
  .figure-img {
    max-width: 100%;
    height: auto;
    border: 1px solid #d2d4dd;
    border-radius: 4px;
    background: #fff;
    padding: 4px;
  }
  .figure-caption {
    font-size: 12px;
    color: #555;
    margin-top: 6px;
  }

  .table-block {
    margin: 18px 0;
  }
  .table-block h3 {
    margin-bottom: 8px;
    font-size: 15px;
  }
  .table-wrapper {
    overflow-x: auto;
    border: 1px solid #d2d4dd;
    border-radius: 4px;
    background: #fafbff;
  }
  table.data-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
  }
  table.data-table th,
  table.data-table td {
    padding: 6px 8px;
    border-bottom: 1px solid #e0e2ea;
    text-align: right;
    white-space: nowrap;
  }
  table.data-table th {
    background: #edf0f9;
    font-weight: 600;
  }
  table.data-table .table-index {
    text-align: left;
    font-weight: 600;
    background: #f5f6fc;
  }

  code {
    background: #f0f2f7;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 90%;
  }

  .download-link {
    display: inline-block;
    margin-top: 6px;
    font-size: 12px;
    color: #1a73e8;
    text-decoration: none;
  }
  .download-link:hover {
    text-decoration: underline;
  }

  .back-to-menu {
    text-align: right;
    margin-top: 12px;
    font-size: 12px;
  }
  .back-to-menu a {
    color: #1a73e8;
    text-decoration: none;
  }
  .back-to-menu a:hover {
    text-decoration: underline;
  }

  .warning { color: #b00020; font-size: 12px; }

  /* =========================
     PRINT / PDF RULES
     ========================= */
  @page {
    size: A4;
    margin: 12mm 12mm 14mm 12mm;
    @bottom-center {
      content: "Página " counter(page) " de " counter(pages);
      font-size: 9px;
      color: #666;
    }
  }

  @media print {
    /* avoid fake “nice shadows” in PDF */
    section, nav { box-shadow: none !important; }

    /* allow long sections to continue */
    section { break-inside: auto; page-break-inside: auto; }

    /* TABLES: keep inside page width */
    .table-wrapper { overflow: visible !important; }

    table.data-table {
      table-layout: fixed !important;
      width: 100% !important;
      font-size: 8.5px !important;
    }

    thead { display: table-header-group; }
    tfoot { display: table-footer-group; }

    table.data-table th,
    table.data-table td {
      white-space: normal !important;
      overflow-wrap: anywhere !important;
      word-break: break-word !important;
      line-height: 1.25 !important;
      text-align: left !important;
      vertical-align: top !important;
    }

    tr { break-inside: avoid; page-break-inside: avoid; }

    code {
      white-space: pre-wrap !important;
      overflow-wrap: anywhere !important;
      word-break: break-word !important;
    }

    .figure-img { max-width: 100% !important; height: auto !important; }
  }
</style>
"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(self.title)}</title>
  {css}
</head>
<body>
  <header>
    <h1>{html.escape(self.title)}</h1>
    <p>{html.escape(self.subtitle)}</p>
  </header>
  <div class="container">
    {nav_html}
    {"\n".join(body_parts)}
  </div>
</body>
</html>
"""


# =============================================================================
# Section builders (NO numbering here)
# =============================================================================

def add_section_eda_overview(report: Report, tables_dir: Path, figs_dir: Path) -> None:
    """Add the EDA overview section to the report."""
    sec = report.add_section("eda-overview", "Integrated EDA Overview")

    label_df = load_table(tables_dir, "label_distribution.csv")
    num_df = load_table(tables_dir, "univariate_numeric_summary.csv")
    cat_df = load_table(tables_dir, "univariate_categorical_summary.csv")
    manifest_df = load_table(tables_dir, "outputs_manifest.csv")

    total_emails = None
    n_labels = None
    if label_df is not None and "count" in label_df.columns:
        try:
            total_emails = int(label_df["count"].sum())
            n_labels = int(label_df.shape[0])
        except Exception:
            pass

    n_numeric = None
    if num_df is not None:
        try:
            n_numeric = int(num_df["column"].nunique()) if "column" in num_df.columns else int(num_df.shape[0])
        except Exception:
            pass

    n_categorical = None
    if cat_df is not None:
        try:
            n_categorical = int(cat_df["column"].nunique()) if "column" in cat_df.columns else int(cat_df.shape[0])
        except Exception:
            pass

    kpi = []
    if total_emails is not None:
        kpi.append(f"<li><strong>Total emails:</strong> {total_emails}</li>")
    if n_labels is not None:
        kpi.append(f"<li><strong>Distinct labels:</strong> {n_labels}</li>")
    if n_numeric is not None:
        kpi.append(f"<li><strong>Numeric columns summarized:</strong> {n_numeric}</li>")
    if n_categorical is not None:
        kpi.append(f"<li><strong>Categorical/text-like columns summarized:</strong> {n_categorical}</li>")

    sec.p(
        "This section provides an integrated map of all exploratory data analysis (EDA) artifacts "
        "generated for SpaPhish. It summarizes where to find tables and figures related to each topic."
    )

    sec.subsection("High-level dataset indicators")
    if kpi:
        sec.raw("<ul>" + "\n".join(kpi) + "</ul>")
    else:
        sec.p("<span class='warning'>Basic indicators could not be derived from summary tables.</span>")

    sec.subsection("Map of EDA artifacts (by topic)")
    eda_map_html = """
    <div class="table-wrapper">
      <table class="data-table">
        <thead>
          <tr>
            <th class="table-index">EDA topic</th>
            <th>Section anchor</th>
            <th>Key tables</th>
            <th>Key figures</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td class="table-index">Data quality (missingness)</td>
            <td><a href="#data-quality">Data Quality</a></td>
            <td><code>missingness_summary.csv</code></td>
            <td><code>missingness_heatmap.png</code>, <code>missingness_barplot.png</code></td>
          </tr>
          <tr>
            <td class="table-index">Label distribution</td>
            <td><a href="#label-distribution">Label Distribution</a></td>
            <td><code>label_distribution.csv</code></td>
            <td><code>label_distribution_bar.png</code></td>
          </tr>
          <tr>
            <td class="table-index">URLs</td>
            <td><a href="#urls">URLs</a></td>
            <td><code>url_statistics.csv</code>, <code>url_domain_frequencies.csv</code>, <code>url_scheme_frequencies.csv</code></td>
            <td><code>url_count_histogram.png</code>, <code>top_url_domains_bar.png</code></td>
          </tr>
          <tr>
            <td class="table-index">Attachments</td>
            <td><a href="#attachments">Attachments</a></td>
            <td><code>attachment_statistics.csv</code>, <code>attachment_type_frequencies.csv</code>, <code>attachment_unit_size_statistics.csv</code></td>
            <td><code>attachment_count_histogram.png</code>, <code>attachment_types_bar.png</code>, <code>attachment_unit_size_histogram.png</code></td>
          </tr>
          <tr>
            <td class="table-index">Length (subject/body)</td>
            <td><a href="#length">Email Length</a></td>
            <td><code>email_length_statistics.csv</code></td>
            <td><code>subject_length_histogram.png</code>, <code>body_length_histogram.png</code>, <code>subject_char_length_histogram.png</code>, <code>body_char_length_histogram.png</code></td>
          </tr>
          <tr>
            <td class="table-index">Routing / hops</td>
            <td><a href="#hops">Routing Complexity</a></td>
            <td><code>hops_statistics.csv</code></td>
            <td><code>hops_count_histogram.png</code></td>
          </tr>
          <tr>
            <td class="table-index">Temporal patterns</td>
            <td><a href="#temporal">Temporal Distribution</a></td>
            <td><code>emails_over_time_year_month.csv</code></td>
            <td><code>emails_over_time_year_month.png</code></td>
          </tr>
          <tr>
            <td class="table-index">Persuasion intensity &amp; reliability</td>
            <td><a href="#pop-reliability">PoP Reliability</a></td>
            <td><code>pop_descriptive_statistics_by_rater.csv</code>, <code>pop_reliability_icc_and_correlations.csv</code>, <code>pop_justifications_overview.csv</code></td>
            <td><code>pop_mean_intensity_by_principle.png</code>, <code>pop_icc_by_principle.png</code></td>
          </tr>
          <tr>
            <td class="table-index">Baselines</td>
            <td><a href="#baseline">Baseline Benchmark</a></td>
            <td><code>baseline_metrics.csv</code></td>
            <td><code>baseline_confusion_logreg.png</code>, <code>baseline_confusion_rf.png</code>, <code>baseline_roc_logreg.png</code>, <code>baseline_roc_rf.png</code></td>
          </tr>
          <tr>
            <td class="table-index">Label sanity checks</td>
            <td><a href="#label-sanity">Label Sanity Check</a></td>
            <td><code>label_sanity_check.csv</code>, <code>label_sanity_anomalies.csv</code></td>
            <td><code>label_sanity_structural_boxplots.png</code>, <code>label_sanity_anomaly_scatter.png</code></td>
          </tr>
          <tr>
            <td class="table-index">Wordclouds</td>
            <td><a href="#wordclouds">Wordclouds</a></td>
            <td>&mdash;</td>
            <td><code>wordcloud_*.png</code></td>
          </tr>
          <tr>
            <td class="table-index">Textual examples</td>
            <td><a href="#textual-examples">Textual Examples</a></td>
            <td><code>examples_by_label.csv</code>, <code>examples_high_pop.csv</code></td>
            <td>&mdash;</td>
          </tr>
          <tr>
            <td class="table-index">Bias analysis</td>
            <td><a href="#bias-analysis">Potential Bias</a></td>
            <td><code>bias_feature_label_association.csv</code>, <code>bias_temporal_label_distribution.csv</code>, <code>bias_bucket_chi2.csv</code>, <code>bias_effect_sizes_label_vs_rest.csv</code></td>
            <td><code>bias_effect_sizes_*_vs_rest.png</code></td>
          </tr>
          <tr>
            <td class="table-index">Value range</td>
            <td><a href="#value-range-validation">Value Range Validation</a></td>
            <td><code>univariate_numeric_summary.csv</code></td>
            <td>&mdash;</td>
          </tr>
          <tr>
            <td class="table-index">Outputs manifest</td>
            <td><a href="#outputs-manifest">Outputs Manifest</a></td>
            <td><code>outputs_manifest.csv</code></td>
            <td>&mdash;</td>
          </tr>
          <tr>
            <td class="table-index">Dataset schema</td>
            <td><a href="#dataset-schema">Dataset Schema</a></td>
            <td><code>dataset_schema_json.json</code></td>
            <td>&mdash;</td>
          </tr>
        </tbody>
      </table>
    </div>
    """
    sec.raw(eda_map_html)

    sec.subsection("Integrated multi-panel dataset snapshot")
    img_html = img_tag(
        figs_dir,
        "data_summary_overview.png",
        "Multi-panel dataset overview (labels, URLs, attachments, length, time, PoP intensity)",
    )
    if img_html:
        sec.figure(
            "data-summary-overview",
            "Multi-panel dataset overview",
            img_html + """
            <p class="figure-caption">
              Multi-panel summary combining label distribution, URL and attachment counts, body length,
              temporal volume (year-month), and mean persuasion intensity by principle.
            </p>
            """,
        )
    else:
        sec.p("<span class='warning'>Integrated overview figure not available.</span>")

    sec.subsection("Auto-generated outputs manifest (preview)")
    if manifest_df is None:
        sec.p("<span class='warning'>outputs_manifest.csv not available.</span>")
    else:
        sec.table(
            "outputs-manifest-preview",
            "Outputs manifest preview",
            table_preview_html(manifest_df, max_rows=20, max_cols=6)
            + f"<div class='table-links'>{download_link('outputs_manifest.csv','Download full outputs_manifest.csv')}</div>",
        )


def add_section_data_quality(report: Report, tables_dir: Path, figs_dir: Path) -> None:
    """Add the Data quality section to the report."""
    sec = report.add_section("data-quality", "Data Quality Assessment (Missingness)")

    miss_df = load_table(tables_dir, "missingness_summary.csv")
    heatmap_img = img_tag(figs_dir, "missingness_heatmap.png", "Missingness heatmap")
    barplot_img = img_tag(figs_dir, "missingness_barplot.png", "Missingness barplot")

    sec.p(
        "This section reports basic data quality diagnostics focused on missing values. "
        "For each column, we compute the number and percentage of missing entries and visualize "
        "the missingness pattern."
    )

    sec.figure(
        "missingness-heatmap",
        "Missingness matrix (heatmap)",
        (heatmap_img or "<p class='warning'>missingness_heatmap.png not available.</p>")
        + """
        <p class="figure-caption">
          Binary heatmap showing which cells are missing versus present (restricted to columns with at least
          one missing value).
        </p>
        """,
    )

    sec.figure(
        "missingness-barplot",
        "Percentage of missing values per column",
        (barplot_img or "<p class='warning'>missingness_barplot.png not available.</p>")
        + """
        <p class="figure-caption">
          Barplot of the percentage of missing values for each column in SpaPhish.
        </p>
        """,
    )

    sec.table(
        "missingness-summary",
        "Missingness summary",
        table_preview_html(miss_df, max_rows=30, max_cols=3)
        + f"<div class='table-links'>{download_link('missingness_summary.csv','Download full missingness_summary.csv')}</div>",
    )


def add_section_label_distribution(report: Report, tables_dir: Path, figs_dir: Path) -> None:
    """Add the Label distribution section to the report."""
    sec = report.add_section("label-distribution", "Label Distribution")

    df = load_table(tables_dir, "label_distribution.csv")
    img_html = img_tag(figs_dir, "label_distribution_bar.png", "Label distribution bar chart")

    sec.p(
        "This section summarizes how many emails in SpaPhish are labeled as phishing versus legitimate "
        "(or other classes, depending on the <code>Label</code> field)."
    )

    sec.figure(
        "label-distribution-bar",
        "Label distribution",
        (img_html or "<p class='warning'>label_distribution_bar.png not available.</p>")
        + "<p class='figure-caption'>Bar chart of the number of emails per label.</p>",
    )

    sec.table(
        "label-distribution-table",
        "Label distribution (counts and percentages)",
        table_preview_html(df, max_rows=20, max_cols=8)
        + f"<div class='table-links'>{download_link('label_distribution.csv','Download full label_distribution.csv')}</div>",
    )


def add_section_urls(report: Report, tables_dir: Path, figs_dir: Path) -> None:
    """Add the URLs section to the report."""
    sec = report.add_section("urls", "URLs")

    stats_df = load_table(tables_dir, "url_statistics.csv")
    domains_df = load_table(tables_dir, "url_domain_frequencies.csv")
    scheme_df = load_table(tables_dir, "url_scheme_frequencies.csv")

    url_count_img = img_tag(figs_dir, "url_count_histogram.png", "Distribution of URL counts")
    domain_bar_img = img_tag(figs_dir, "top_url_domains_bar.png", "Top URL domains")

    sec.p(
        "URL data provides insight into sender infrastructure, domain diversity, and differences between "
        "phishing and benign emails."
    )

    sec.figure(
        "url-count-histogram",
        "URL count distribution",
        (url_count_img or "<p class='warning'>url_count_histogram.png not available.</p>")
        + "<p class='figure-caption'>Histogram of the number of URLs per email.</p>",
    )

    sec.table(
        "url-statistics",
        "URL statistics",
        table_preview_html(stats_df, max_rows=12, max_cols=10)
        + f"<div class='table-links'>{download_link('url_statistics.csv','Download full url_statistics.csv')}</div>",
    )

    sec.table(
        "url-scheme-frequencies",
        "URL scheme frequencies",
        table_preview_html(scheme_df, max_rows=12, max_cols=10)
        + f"<div class='table-links'>{download_link('url_scheme_frequencies.csv','Download full url_scheme_frequencies.csv')}</div>",
    )

    sec.figure(
        "top-url-domains",
        "Top URL domains",
        (domain_bar_img or "<p class='warning'>top_url_domains_bar.png not available.</p>")
        + "<p class='figure-caption'>Ranking of most frequently appearing domains across all emails.</p>",
    )

    sec.table(
        "url-domain-frequencies",
        "Domain frequencies",
        table_preview_html(domains_df, max_rows=15, max_cols=10)
        + f"<div class='table-links'>{download_link('url_domain_frequencies.csv','Download full url_domain_frequencies.csv')}</div>",
    )


def add_section_attachments(report: Report, tables_dir: Path, figs_dir: Path) -> None:
    """Add the Attachments section to the report."""
    sec = report.add_section("attachments", "Attachments")

    stats_df = load_table(tables_dir, "attachment_statistics.csv")
    types_df = load_table(tables_dir, "attachment_type_frequencies.csv")
    unit_stats_df = load_table(tables_dir, "attachment_unit_size_statistics.csv")

    count_img = img_tag(figs_dir, "attachment_count_histogram.png", "Histogram of attachments per email")
    types_img = img_tag(figs_dir, "attachment_types_bar.png", "Attachment types bar chart")
    unit_size_img = img_tag(figs_dir, "attachment_unit_size_histogram.png", "Histogram of individual attachment sizes")

    sec.p(
        "Attachments often carry malicious payloads (e.g., executable files, macro-enabled documents). "
        "This section characterizes how frequently attachments appear, which types are most common, and "
        "how large individual attachment files tend to be."
    )

    sec.figure(
        "attachment-count-histogram",
        "Attachment count per email",
        (count_img or "<p class='warning'>attachment_count_histogram.png not available.</p>")
        + "<p class='figure-caption'>Histogram of number of attachments per email.</p>",
    )

    sec.table(
        "attachment-statistics",
        "Attachment statistics per email",
        table_preview_html(stats_df, max_rows=15, max_cols=12)
        + f"<div class='table-links'>{download_link('attachment_statistics.csv','Download full attachment_statistics.csv')}</div>",
    )

    sec.figure(
        "attachment-types",
        "Attachment types",
        (types_img or "<p class='warning'>attachment_types_bar.png not available.</p>")
        + "<p class='figure-caption'>Most frequent attachment types or extensions in SpaPhish.</p>",
    )

    sec.table(
        "attachment-type-frequencies",
        "Attachment type frequencies",
        table_preview_html(types_df, max_rows=20, max_cols=10)
        + f"<div class='table-links'>{download_link('attachment_type_frequencies.csv','Download full attachment_type_frequencies.csv')}</div>",
    )

    sec.figure(
        "attachment-unit-size-histogram",
        "Individual attachment sizes",
        (unit_size_img or "<p class='warning'>attachment_unit_size_histogram.png not available.</p>")
        + "<p class='figure-caption'>Histogram of individual attachment sizes.</p>",
    )

    sec.table(
        "attachment-unit-size-statistics",
        "Individual attachment size statistics",
        table_preview_html(unit_stats_df, max_rows=15, max_cols=12)
        + f"<div class='table-links'>{download_link('attachment_unit_size_statistics.csv','Download full attachment_unit_size_statistics.csv')}</div>",
    )


def add_section_length(report: Report, tables_dir: Path, figs_dir: Path) -> None:
    """Add the Length section to the report."""
    sec = report.add_section("length", "Email Length (Subject and Body)")

    length_df = load_table(tables_dir, "email_length_statistics.csv")

    subj_words_img = img_tag(figs_dir, "subject_length_histogram.png", "Histogram of subject length (words)")
    body_words_img = img_tag(figs_dir, "body_length_histogram.png", "Histogram of body length (words)")
    subj_chars_img = img_tag(figs_dir, "subject_char_length_histogram.png", "Histogram of subject length (characters)")
    body_chars_img = img_tag(figs_dir, "body_char_length_histogram.png", "Histogram of body length (characters)")

    sec.p(
        "Beyond content, email length is a basic but informative structural attribute. SpaPhish reports "
        "both character-level and word-level length for subject and body."
    )

    sec.table(
        "email-length-statistics",
        "Subject and body length statistics",
        table_preview_html(length_df, max_rows=25, max_cols=12)
        + f"<div class='table-links'>{download_link('email_length_statistics.csv','Download full email_length_statistics.csv')}</div>",
    )

    grid = f"""
    <div class="figure-grid">
      <div class="figure-block">
        {subj_words_img or "<p class='warning'>subject_length_histogram.png not available.</p>"}
        <p class="figure-caption">Subject length (words).</p>
      </div>
      <div class="figure-block">
        {body_words_img or "<p class='warning'>body_length_histogram.png not available.</p>"}
        <p class="figure-caption">Body length (words).</p>
      </div>
      <div class="figure-block">
        {subj_chars_img or "<p class='warning'>subject_char_length_histogram.png not available.</p>"}
        <p class="figure-caption">Subject length (characters).</p>
      </div>
      <div class="figure-block">
        {body_chars_img or "<p class='warning'>body_char_length_histogram.png not available.</p>"}
        <p class="figure-caption">Body length (characters).</p>
      </div>
    </div>
    """
    sec.raw(grid)


def add_section_hops(report: Report, tables_dir: Path, figs_dir: Path) -> None:
    """Add the Hops section to the report."""
    sec = report.add_section("hops", "Routing Complexity (Hops Count)")

    hops_df = load_table(tables_dir, "hops_statistics.csv")
    hops_img = img_tag(figs_dir, "hops_count_histogram.png", "Histogram of hops count per email")

    sec.p(
        "The <code>hops_count</code> field captures how many intermediate hops are present in the "
        "email delivery path, providing a coarse measure of routing complexity."
    )

    sec.figure(
        "hops-count-histogram",
        "Hops count per email",
        (hops_img or "<p class='warning'>hops_count_histogram.png not available.</p>")
        + "<p class='figure-caption'>Histogram of the number of hops observed per email.</p>",
    )

    sec.table(
        "hops-statistics",
        "Hops count statistics",
        table_preview_html(hops_df, max_rows=25, max_cols=12)
        + f"<div class='table-links'>{download_link('hops_statistics.csv','Download full hops_statistics.csv')}</div>",
    )


def add_section_temporal(report: Report, tables_dir: Path, figs_dir: Path) -> None:
    """Add the Temporal section to the report."""
    sec = report.add_section("temporal", "Temporal Distribution")

    temp_df = load_table(tables_dir, "emails_over_time_year_month.csv")
    temp_img = img_tag(figs_dir, "emails_over_time_year_month.png", "Emails over time (year-month)")

    sec.p(
        "Temporal patterns help identify waves or campaigns of phishing attacks. This section shows "
        "how many emails are present per year-month, based on the <code>date</code> field."
    )

    sec.figure(
        "emails-over-time",
        "Emails over time",
        (temp_img or "<p class='warning'>emails_over_time_year_month.png not available.</p>")
        + "<p class='figure-caption'>Monthly volume of emails aggregated by year-month.</p>",
    )

    sec.table(
        "emails-over-time-table",
        "Emails over time (year-month)",
        table_preview_html(temp_df, max_rows=24, max_cols=10)
        + f"<div class='table-links'>{download_link('emails_over_time_year_month.csv','Download full emails_over_time_year_month.csv')}</div>",
    )


def add_section_pop_reliability(report: Report, tables_dir: Path, figs_dir: Path) -> None:
    """Add the POP reliability section to the report."""
    sec = report.add_section("pop-reliability", "Persuasion Intensity, Justifications and Inter-Rater Reliability")

    desc_df = load_table(tables_dir, "pop_descriptive_statistics_by_rater.csv")
    rel_df = load_table(tables_dir, "pop_reliability_icc_and_correlations.csv")
    justif_df = load_table(tables_dir, "pop_justifications_overview.csv")

    mean_img = img_tag(figs_dir, "pop_mean_intensity_by_principle.png", "Mean PoP intensity by principle")
    icc_img = img_tag(figs_dir, "pop_icc_by_principle.png", "ICC by principle")

    sec.p(
        "SpaPhish includes manual annotations of persuasion intensity for multiple principles of influence. "
        "Three independent raters scored each email, enabling descriptive statistics and inter-rater reliability."
    )

    sec.figure(
        "pop-mean-by-principle",
        "Mean PoP intensity by principle",
        (mean_img or "<p class='warning'>pop_mean_intensity_by_principle.png not available.</p>")
        + "<p class='figure-caption'>Mean persuasion intensity for each principle.</p>",
    )

    sec.figure(
        "pop-icc-by-principle",
        "Inter-rater reliability per principle",
        (icc_img or "<p class='warning'>pop_icc_by_principle.png not available.</p>")
        + "<p class='figure-caption'>ICC(2,1) per persuasion principle.</p>",
    )

    sec.table(
        "pop-descriptive-by-rater",
        "PoP descriptive statistics by rater",
        table_preview_html(desc_df, max_rows=12, max_cols=12)
        + f"<div class='table-links'>{download_link('pop_descriptive_statistics_by_rater.csv','Download full pop_descriptive_statistics_by_rater.csv')}</div>",
    )

    sec.table(
        "pop-reliability",
        "PoP inter-rater reliability",
        table_preview_html(rel_df, max_rows=12, max_cols=12)
        + f"<div class='table-links'>{download_link('pop_reliability_icc_and_correlations.csv','Download full pop_reliability_icc_and_correlations.csv')}</div>",
    )

    # NOTE: This is NOT "Table S1" anymore; the numbering is automatic.
    sec.table(
        "pop-justifications-overview",
        "Coverage of PoP justification texts",
        table_preview_html(justif_df, max_rows=12, max_cols=12)
        + f"<div class='table-links'>{download_link('pop_justifications_overview.csv','Download full pop_justifications_overview.csv')}</div>",
    )


def add_section_pop_vs_label(report: Report, tables_dir: Path, figs_dir: Path) -> None:
    """Add the POP vs label section to the report."""
    sec = report.add_section("pop-vs-label", "Association Between Persuasion Intensity and Labels")

    assoc_df = load_table(tables_dir, "pop_label_association_tests.csv")

    sec.p(
        "SpaPhish supports analysis of how persuasion intensity relates to the final label (e.g., phishing vs legitimate). "
        "This section summarizes association tests and provides per-principle boxplots when available."
    )

    sec.table(
        "pop-label-association-tests",
        "Association tests between PoP intensity and Label",
        table_preview_html(assoc_df, max_rows=15, max_cols=12)
        + f"<div class='table-links'>{download_link('pop_label_association_tests.csv','Download full pop_label_association_tests.csv')}</div>",
    )

    principle_filenames = {
        "authority": "pop_intensity_by_label_authority.png",
        "social_proof": "pop_intensity_by_label_social_proof.png",
        "liking_similarity_deception": "pop_intensity_by_label_liking_similarity_deception.png",
        "commitment_integrity_reciprocation": "pop_intensity_by_label_commitment_integrity_reciprocation.png",
        "distraction": "pop_intensity_by_label_distraction.png",
    }

    cards = []
    for principle, fname in principle_filenames.items():
        im = img_tag(figs_dir, fname, f"{principle} intensity by label")
        if not im:
            continue
        cards.append(f"""
          <div class="figure-block">
            {im}
            <p class="figure-caption">Boxplot of {html.escape(principle.replace('_',' '))} intensity across labels.</p>
          </div>
        """)
    if cards:
        sec.raw('<div class="figure-grid">' + "\n".join(cards) + "</div>")
    else:
        sec.p("<span class='warning'>No PoP-vs-label boxplots available.</span>")


def add_section_baseline(report: Report, tables_dir: Path, figs_dir: Path) -> None:
    """Add the Baseline section to the report."""
    sec = report.add_section("baseline", "Baseline Text Classification Benchmark")

    metrics_df = load_table(tables_dir, "baseline_metrics.csv")

    cm_logreg_img = img_tag(figs_dir, "baseline_confusion_logreg.png", "Confusion matrix (Logistic Regression)")
    cm_rf_img = img_tag(figs_dir, "baseline_confusion_rf.png", "Confusion matrix (Random Forest)")
    roc_logreg_img = img_tag(figs_dir, "baseline_roc_logreg.png", "ROC curve (Logistic Regression)")
    roc_rf_img = img_tag(figs_dir, "baseline_roc_rf.png", "ROC curve (Random Forest)")

    sec.p(
        "To show SpaPhish is directly usable for supervised phishing detection, we provide a baseline benchmark "
        "using TF–IDF representations of subject+body and two standard classifiers: Logistic Regression and Random Forest."
    )

    sec.table(
        "baseline-metrics",
        "Baseline performance metrics",
        table_preview_html(metrics_df, max_rows=12, max_cols=12)
        + f"<div class='table-links'>{download_link('baseline_metrics.csv','Download full baseline_metrics.csv')}</div>",
    )

    grid = f"""
    <div class="figure-grid">
      <div class="figure-block">
        {cm_logreg_img or "<p class='warning'>baseline_confusion_logreg.png not available.</p>"}
        <p class="figure-caption">Confusion matrix – Logistic Regression.</p>
      </div>
      <div class="figure-block">
        {cm_rf_img or "<p class='warning'>baseline_confusion_rf.png not available.</p>"}
        <p class="figure-caption">Confusion matrix – Random Forest.</p>
      </div>
    </div>
    """
    sec.raw(grid)

    roc_cards = []
    if roc_logreg_img:
        roc_cards.append(f"<div class='figure-block'>{roc_logreg_img}<p class='figure-caption'>ROC – Logistic Regression.</p></div>")
    if roc_rf_img:
        roc_cards.append(f"<div class='figure-block'>{roc_rf_img}<p class='figure-caption'>ROC – Random Forest.</p></div>")
    if roc_cards:
        sec.raw('<div class="figure-grid">' + "\n".join(roc_cards) + "</div>")


def add_section_label_sanity(report: Report, tables_dir: Path, figs_dir: Path) -> None:
    """Add the Label sanity section to the report."""
    sec = report.add_section("label-sanity", "Label Sanity Check")

    check_df = load_table(tables_dir, "label_sanity_check.csv")
    anomalies_df = load_table(tables_dir, "label_sanity_anomalies.csv")

    boxplots_img = img_tag(figs_dir, "label_sanity_structural_boxplots.png", "Structural feature boxplots by label")
    anomaly_scatter_img = img_tag(figs_dir, "label_sanity_anomaly_scatter.png", "Label anomaly scatter")

    sec.p(
        "This section examines internal consistency of the <code>Label</code> field with respect to structural features. "
        "It compares distributions across labels and highlights potential anomalies for manual review."
    )

    sec.figure(
        "label-sanity-structural-boxplots",
        "Structural features by label",
        (boxplots_img or "<p class='warning'>label_sanity_structural_boxplots.png not available.</p>")
        + "<p class='figure-caption'>Boxplots of selected structural features stratified by label.</p>",
    )

    sec.table(
        "label-sanity-check",
        "Label-wise structural summary",
        table_preview_html(check_df, max_rows=12, max_cols=12)
        + f"<div class='table-links'>{download_link('label_sanity_check.csv','Download full label_sanity_check.csv')}</div>",
    )

    sec.figure(
        "label-sanity-anomaly-scatter",
        "Anomalous label assignments",
        (anomaly_scatter_img or "<p class='warning'>label_sanity_anomaly_scatter.png not available.</p>")
        + "<p class='figure-caption'>Scatter plot highlighting emails flagged as label anomalies.</p>",
    )

    sec.table(
        "label-sanity-anomalies",
        "Flagged anomalous emails",
        table_preview_html(anomalies_df, max_rows=20, max_cols=12)
        + f"<div class='table-links'>{download_link('label_sanity_anomalies.csv','Download full label_sanity_anomalies.csv')}</div>",
    )


def add_section_wordclouds(report: Report, figs_dir: Path) -> None:
    """Add the Wordclouds section to the report."""
    sec = report.add_section("wordclouds", "Wordclouds")

    sec.p(
        "Visual summaries of subject+body content. Includes a global wordcloud and any per-label or "
        "principle×label wordclouds available under <code>output/figures/</code>."
    )

    global_img = img_tag(figs_dir, "wordcloud_global.png", "Global wordcloud (subject + body)")
    sec.figure(
        "wordcloud-global",
        "Global wordcloud",
        (global_img or "<p class='warning'>wordcloud_global.png not available.</p>")
        + "<p class='figure-caption'>Global wordcloud built from subject+body.</p>",
    )

    # Thumbnail subsets
    label_paths = sorted(figs_dir.glob("wordcloud_label_*.png"))
    pop_label_paths = sorted([p for p in figs_dir.glob("wordcloud_*_label_*.png") if not p.name.startswith("wordcloud_label_")])

    def grid_html(paths: List[Path], limit: int = 12) -> str:
        """Perform the Grid HTML operation."""
        if not paths:
            return "<p class='warning'>No images available.</p>"
        cards = []
        for p in paths[:limit]:
            im = img_tag(figs_dir, p.name, p.name)
            if im:
                cards.append(f"<div class='figure-block'>{im}</div>")
        return '<div class="figure-grid">' + "\n".join(cards) + "</div>"

    sec.subsection("Per-label wordclouds (thumbnail subset)")
    sec.raw(grid_html(label_paths, limit=12))

    sec.subsection("Principle × Label wordclouds (thumbnail subset)")
    sec.raw(grid_html(pop_label_paths, limit=12))


def add_section_textual_examples(report: Report, tables_dir: Path) -> None:
    """Add the Textual examples section to the report."""
    sec = report.add_section("textual-examples", "Textual Examples")

    by_label_df = load_table(tables_dir, "examples_by_label.csv")
    high_pop_df = load_table(tables_dir, "examples_high_pop.csv")

    sec.p(
        "This section provides representative email examples for qualitative inspection. "
        "Body content is shown as excerpts to keep the report compact."
    )

    sec.table(
        "examples-by-label",
        "Examples by label",
        table_preview_html(by_label_df, max_rows=12, max_cols=12)
        + f"<div class='table-links'>{download_link('examples_by_label.csv','Download full examples_by_label.csv')}</div>",
    )

    sec.table(
        "examples-high-pop",
        "High-intensity persuasion examples",
        table_preview_html(high_pop_df, max_rows=20, max_cols=12)
        + f"<div class='table-links'>{download_link('examples_high_pop.csv','Download full examples_high_pop.csv')}</div>",
    )


def add_section_bias_analysis(report: Report, tables_dir: Path, figs_dir: Path) -> None:
    """Add the Bias analysis section to the report."""
    sec = report.add_section("bias-analysis", "Potential Bias Analysis")

    assoc_df = load_table(tables_dir, "bias_feature_label_association.csv")
    temporal_counts_df = load_table(tables_dir, "bias_temporal_label_distribution.csv")
    temporal_props_df = load_table(tables_dir, "bias_temporal_label_proportions.csv")
    chi2_df = load_table(tables_dir, "bias_bucket_chi2.csv")
    effects_df = load_table(tables_dir, "bias_effect_sizes_label_vs_rest.csv")

    effect_img_html = ""
    for p in figs_dir.glob("bias_effect_sizes_*_vs_rest.png"):
        effect_img_html = img_tag(figs_dir, p.name, "Effect sizes for label vs rest")
        if effect_img_html:
            break

    sec.p(
        "This section explores potential biases by examining how <code>Label</code> relates to structural "
        "and persuasion-related features, and how label composition evolves over time."
    )

    sec.table(
        "bias-feature-label-association",
        "Association between label and numeric features",
        table_preview_html(assoc_df, max_rows=15, max_cols=12)
        + f"<div class='table-links'>{download_link('bias_feature_label_association.csv','Download full bias_feature_label_association.csv')}</div>",
    )

    sec.table(
        "bias-temporal-label-distribution",
        "Temporal evolution of label distribution",
        table_preview_html(temporal_counts_df, max_rows=12, max_cols=12)
        + "<div class='table-links'>"
          + download_link("bias_temporal_label_distribution.csv","Download bias_temporal_label_distribution.csv")
          + " "
          + download_link("bias_temporal_label_proportions.csv","Download bias_temporal_label_proportions.csv")
          + "</div>",
    )

    sec.table(
        "bias-bucket-chi2",
        "Bucketed chi-square tests",
        table_preview_html(chi2_df, max_rows=12, max_cols=12)
        + f"<div class='table-links'>{download_link('bias_bucket_chi2.csv','Download full bias_bucket_chi2.csv')}</div>",
    )

    sec.figure(
        "bias-effect-sizes-figure",
        "Effect sizes for one label versus the rest",
        (effect_img_html or "<p class='warning'>No bias_effect_sizes_*_vs_rest.png figure available.</p>")
        + "<p class='figure-caption'>Absolute standardized mean differences (SMD) across numeric features.</p>",
    )

    sec.table(
        "bias-effect-sizes-table",
        "Effect sizes table",
        table_preview_html(effects_df, max_rows=20, max_cols=12)
        + f"<div class='table-links'>{download_link('bias_effect_sizes_label_vs_rest.csv','Download full bias_effect_sizes_label_vs_rest.csv')}</div>",
    )


def add_section_value_range(report: Report, tables_dir: Path) -> None:
    """Add the Value range section to the report."""
    sec = report.add_section("value-range-validation", "Value Range Validation")

    num_df = load_table(tables_dir, "univariate_numeric_summary.csv")
    if num_df is None:
        sec.p("<span class='warning'>univariate_numeric_summary.csv not available.</span>")
        return

    df = num_df.copy()
    if "column_name" in df.columns:
        df = df.set_index("column_name")

    high_missing = None
    high_outliers = None

    if "pct_missing" in df.columns:
        high_missing = df[df["pct_missing"] > 20.0].sort_values("pct_missing", ascending=False)
    if "pct_outliers_iqr" in df.columns:
        high_outliers = df[df["pct_outliers_iqr"] > 5.0].sort_values("pct_outliers_iqr", ascending=False)

    sec.p(
        "This section summarizes simple value-range checks for numeric variables based on "
        "<code>univariate_numeric_summary.csv</code>, focusing on missingness and IQR-based outliers."
    )

    sec.subsection("Columns with high missingness (> 20%)")
    if high_missing is not None and not high_missing.empty:
        sec.table("high-missingness", "Numeric columns with high missingness", table_preview_html(high_missing, max_rows=25, max_cols=12))
    else:
        sec.p("No numeric columns exceed the 20% missingness threshold.")

    sec.subsection("Columns with many IQR-based outliers (> 5%)")
    if high_outliers is not None and not high_outliers.empty:
        sec.table("high-outliers", "Numeric columns with many IQR outliers", table_preview_html(high_outliers, max_rows=25, max_cols=12))
    else:
        sec.p("No numeric columns exceed the 5% IQR-based outlier threshold.")


def add_section_outputs_manifest(report: Report, tables_dir: Path) -> None:
    """Add the Outputs manifest section to the report."""
    sec = report.add_section("outputs-manifest", "Outputs Manifest")

    manifest_df = load_table(tables_dir, "outputs_manifest.csv")
    sec.p(
        "This section exposes the automatically generated manifest of SpaPhish artifacts under <code>output/</code>. "
        "Each row corresponds to a table or figure with basic metadata."
    )

    sec.table(
        "outputs-manifest-full",
        "Outputs manifest",
        table_preview_html(manifest_df, max_rows=40, max_cols=10)
        + f"<div class='table-links'>{download_link('outputs_manifest.csv','Download full outputs_manifest.csv')}</div>",
    )


def add_section_dataset_schema(report: Report, tables_dir: Path) -> None:
    """Add the Dataset schema section to the report."""
    sec = report.add_section("dataset-schema", "Dataset Schema (JSON overview)")

    schema = load_dataset_schema_json(tables_dir)
    schema_df = None

    if schema is None:
        sec.p("<span class='warning'>dataset_schema_json.json not available.</span>")
        return

    fields = []
    if isinstance(schema, dict):
        if isinstance(schema.get("fields"), list):
            fields = schema["fields"]
        elif isinstance(schema.get("columns"), list):
            fields = schema["columns"]
    elif isinstance(schema, list):
        fields = schema

    rows = []
    for f in fields:
        if not isinstance(f, dict):
            continue
        name = f.get("name") or f.get("column") or ""
        dtype = f.get("type") or f.get("dtype") or ""
        desc = f.get("description") or f.get("doc") or ""
        nullable = f.get("nullable")
        nullable_str = "yes" if nullable is True else ("no" if nullable is False else "")
        rows.append({"column": str(name), "type": str(dtype), "nullable": nullable_str, "description": str(desc)})

    if rows:
        schema_df = pd.DataFrame(rows)

    sec.p(
        "This section summarizes the <code>dataset_schema_json.json</code> file used to document SpaPhish columns "
        "(names, types, nullability, descriptions)."
    )

    sec.table(
        "dataset-schema-table",
        "Column-level schema",
        table_preview_html(schema_df, max_rows=40, max_cols=4)
        + f"<div class='table-links'>{download_link('dataset_schema_json.json','Download full dataset_schema_json.json')}</div>",
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    # This script lives in reporting/, so the project root is one level up.
    """Run the HTML report generator and return an exit code."""
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    tables_dir = output_dir / "tables"
    figs_dir = output_dir / "figures"

    if not output_dir.exists():
        print(f"[ERROR] output/ not found: {output_dir}")
        return 1
    if not tables_dir.exists() or not figs_dir.exists():
        print("[ERROR] output/tables/ or output/figures/ not found. Run analyze_spaphish_dataset.py first.")
        return 1

    try:
        from weasyprint import HTML
    except ImportError:
        print("[ERROR] WeasyPrint not installed. Install: pip install weasyprint")
        return 1

    report = Report(
        title="SpaPhish Dataset – Descriptive Report",
        subtitle="Static HTML/PDF report generated from pre-computed tables and figures under output/."
    )

    # Build report (NO numbering here)
    add_section_eda_overview(report, tables_dir, figs_dir)
    add_section_data_quality(report, tables_dir, figs_dir)
    add_section_label_distribution(report, tables_dir, figs_dir)
    add_section_urls(report, tables_dir, figs_dir)
    add_section_attachments(report, tables_dir, figs_dir)
    add_section_length(report, tables_dir, figs_dir)
    add_section_hops(report, tables_dir, figs_dir)
    add_section_temporal(report, tables_dir, figs_dir)
    add_section_pop_reliability(report, tables_dir, figs_dir)
    add_section_pop_vs_label(report, tables_dir, figs_dir)
    add_section_baseline(report, tables_dir, figs_dir)
    add_section_label_sanity(report, tables_dir, figs_dir)
    add_section_wordclouds(report, figs_dir)
    add_section_textual_examples(report, tables_dir)
    add_section_bias_analysis(report, tables_dir, figs_dir)
    add_section_value_range(report, tables_dir)
    add_section_outputs_manifest(report, tables_dir)
    add_section_dataset_schema(report, tables_dir)

    html_doc = report.render_html()

    html_path = output_dir / "index_updated.html"
    html_path.write_text(html_doc, encoding="utf-8")
    print(f"[OK] HTML written: {html_path}")

    pdf_path = output_dir / "SpaPhish_Report_UPDATED.pdf"
    try:
        # Generate PDF from file to resolve relative resources correctly
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        return 1

    print(f"[OK] PDF written:  {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
