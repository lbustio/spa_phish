"""
report_pdf.py

SpaPhish PDF report generator using ReportLab.

Builds a well-formed, navigable, styled PDF directly (no HTML conversion).
  - Letter page size
  - Table of contents menu + PDF outline + "Back to menu" after each section
  - Section-scoped figure/table numbering (Figure 2.1, Table 2.1, etc.)
  - Narrative text per section in English
  - Captions sourced from outputs_manifest.csv with robust fallbacks (never "nan")
  - EPS figures are converted to PNG via Ghostscript before embedding

Prerequisites:
  Run analysis/analyze_dataset.py first to populate output/tables/ and output/figures/.
  Ghostscript must be installed and on PATH for EPS figure support.

Usage:
  python reporting/report_pdf.py [--output-dir output] [--pdf SpaPhish_Report.pdf]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


# -----------------------------
# Configuration
# -----------------------------

ACCENT = colors.HexColor("#1F5AA6")  # muted blue
TEXT = colors.HexColor("#222222")
LIGHT_BG = colors.HexColor("#F6F8FB")
GRID = colors.HexColor("#D6DCE6")

DEFAULT_LEFT_RIGHT_MARGIN = 0.75 * inch
DEFAULT_TOP_MARGIN = 0.75 * inch
DEFAULT_BOTTOM_MARGIN = 0.75 * inch

# Candidates where a manifest might exist (based on typical layouts)
MANIFEST_CANDIDATES = [
    "outputs_manifest.csv",
    os.path.join("tables", "outputs_manifest.csv"),
    os.path.join("output", "outputs_manifest.csv"),
    os.path.join("output", "tables", "outputs_manifest.csv"),
]

# Tables and figures roots inside output_dir
TABLES_DIRNAME = "tables"
FIGURES_DIRNAME = "figures"

# Which figure formats we try to embed directly
EMBED_FIG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}  # pdf embedding of PDF is non-trivial in reportlab
EPS_EXTS = {".eps"}


# -----------------------------
# Helpers: robust text
# -----------------------------

def is_nan_like(x: Any) -> bool:
    """Return True when a value should be treated as empty or missing."""
    if x is None:
        return True
    s = str(x).strip().lower()
    return s == "" or s == "nan" or s == "none" or s == "null"


def humanize_name(name: str) -> str:
    # label_distribution_bar -> Label distribution bar
    """Convert a machine-style identifier into a human-readable label."""
    s = name.replace("_", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s[:1].upper() + s[1:]


def safe_escape(s: str) -> str:
    """Escape HTML-sensitive characters so text can be embedded safely in ReportLab markup."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


# -----------------------------
# Manifest I/O
# -----------------------------

@dataclass(frozen=True)
class ManifestItem:
    name: str
    path_rel: str
    kind: str  # "table" or "figure"
    fmt: str   # "csv", "png", "eps", ...
    description: str


def load_manifest(output_dir: Path) -> Dict[str, ManifestItem]:
    """Load the generated outputs manifest from the report output directory."""
    found: Optional[Path] = None
    for cand in MANIFEST_CANDIDATES:
        p = (output_dir / cand).resolve()
        if p.exists() and p.is_file():
            found = p
            break

    if not found:
        # No manifest is not fatal; we will still build from directory scan.
        return {}

    items: Dict[str, ManifestItem] = {}
    with found.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # Expect columns like: name,path_rel,kind,format,description
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            items[name] = ManifestItem(
                name=name,
                path_rel=(row.get("path_rel") or "").strip(),
                kind=(row.get("kind") or "").strip(),
                fmt=(row.get("format") or "").strip(),
                description=(row.get("description") or "").strip(),
            )
    return items


# -----------------------------
# EPS conversion (Ghostscript)
# -----------------------------

def find_ghostscript_executable() -> Optional[str]:
    """Return the Ghostscript executable name if one is available."""
    for exe in ("gswin64c", "gswin32c", "gs"):
        if shutil.which(exe):
            return exe
    return None


def convert_eps_to_png(eps_path: Path, png_path: Path, dpi: int = 300) -> None:
    """Convert an EPS figure to PNG using Ghostscript."""
    gs = find_ghostscript_executable()
    if not gs:
        raise RuntimeError(
            "Ghostscript not found. Install it (conda-forge ghostscript) "
            "or export figures as PNG/JPG instead of EPS."
        )

    png_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        gs,
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=pngalpha",
        f"-r{dpi}",
        f"-sOutputFile={str(png_path)}",
        str(eps_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not png_path.exists():
        raise RuntimeError(
            "EPS->PNG conversion failed.\n"
            f"EPS: {eps_path}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR:\n{proc.stderr.strip()}"
        )


# -----------------------------
# Styles
# -----------------------------

def build_styles() -> Dict[str, ParagraphStyle]:
    """Create the paragraph styles used by the PDF report."""
    base = getSampleStyleSheet()

    styles: Dict[str, ParagraphStyle] = {}

    styles["Title"] = ParagraphStyle(
        "Title",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=ACCENT,
        spaceAfter=14,
    )

    styles["Subtitle"] = ParagraphStyle(
        "Subtitle",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=TEXT,
        spaceAfter=10,
    )

    styles["H1"] = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=ACCENT,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    styles["Body"] = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=TEXT,
        spaceAfter=8,
    )

    styles["Caption"] = ParagraphStyle(
        "Caption",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=TEXT,
        spaceBefore=6,
        spaceAfter=10,
    )

    styles["Small"] = ParagraphStyle(
        "Small",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        textColor=TEXT,
        spaceAfter=6,
    )

    styles["MenuItem"] = ParagraphStyle(
        "MenuItem",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=TEXT,
        spaceAfter=6,
    )

    styles["BackLink"] = ParagraphStyle(
        "BackLink",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        textColor=ACCENT,
        spaceBefore=10,
        spaceAfter=12,
    )

    styles["TOCHeading"] = ParagraphStyle(
        "TOCHeading",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=ACCENT,
        spaceAfter=8,
    )

    styles["TOCEntry"] = ParagraphStyle(
        "TOCEntry",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=13,
        textColor=TEXT,
    )

    styles["TableCell"] = ParagraphStyle(
        "TableCell",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        textColor=TEXT,
        wordWrap="CJK",
        splitLongWords=1,
    )

    styles["TableHeader"] = ParagraphStyle(
        "TableHeader",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.white,
        wordWrap="CJK",
        splitLongWords=1,
    )

    return styles


# -----------------------------
# Numbering (section-scoped)
# -----------------------------

class Numbering:
    def __init__(self) -> None:
        """Perform the Init operation."""
        self.section_no = 0
        self.fig_no = 0
        self.tbl_no = 0

    def new_section(self) -> int:
        """Perform the New section operation."""
        self.section_no += 1
        self.fig_no = 0
        self.tbl_no = 0
        return self.section_no

    def next_figure(self) -> str:
        """Perform the Next figure operation."""
        self.fig_no += 1
        return f"{self.section_no}.{self.fig_no}"

    def next_table(self) -> str:
        """Perform the Next table operation."""
        self.tbl_no += 1
        return f"{self.section_no}.{self.tbl_no}"


# -----------------------------
# DocTemplate with outline + TOC
# -----------------------------

class SpaPhishDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs: Any):
        """Perform the Init operation."""
        super().__init__(filename, **kwargs)
        self._last_heading_key: Optional[str] = None

    def afterFlowable(self, flowable: Any) -> None:
        # Collect TOC entries from H1 paragraphs
        """Perform the AfterFlowable operation."""
        if isinstance(flowable, Paragraph) and flowable.style.name == "H1":
            text = flowable.getPlainText()
            key = getattr(flowable, "_bookmark_key", None)
            if key:
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=0, closed=False)
            self.notify("TOCEntry", (0, text, self.page))


# -----------------------------
# Table building: fit to page
# -----------------------------

def string_width(s: str, font_name: str, font_size: float) -> float:
    """Measure the rendered width of a string for the given font settings."""
    try:
        return pdfmetrics.stringWidth(s, font_name, font_size)
    except Exception:
        return len(s) * (font_size * 0.55)


def compute_col_widths(
    data: List[List[Any]],
    avail_width: float,
    font_name: str,
    font_size: float,
    max_rows_probe: int = 50,
    min_col_w: float = 44.0,
    max_col_w: float = 260.0,
) -> List[float]:
    """Compute table column widths that fit the available page width."""
    if not data:
        return []

    ncols = max(len(r) for r in data)
    widths = [min_col_w] * ncols

    probe = data[: min(len(data), max_rows_probe)]
    for c in range(ncols):
        w = min_col_w
        for r in probe:
            if c >= len(r):
                continue
            s = "" if r[c] is None else str(r[c])
            s = re.sub(r"\s+", " ", s).strip()
            w = max(w, min(max_col_w, string_width(s, font_name, font_size) + 14))
        widths[c] = w

    total = sum(widths)
    if total <= avail_width:
        return widths

    # Scale down proportionally but keep minimums
    scale = avail_width / total
    scaled = [max(min_col_w, w * scale) for w in widths]

    # If still too wide due to min constraints, squeeze further uniformly
    total2 = sum(scaled)
    if total2 > avail_width:
        overflow = total2 - avail_width
        reducible = [w - min_col_w for w in scaled]
        reducible_total = sum(max(0.0, x) for x in reducible)
        if reducible_total > 0:
            for i in range(len(scaled)):
                if reducible[i] <= 0:
                    continue
                scaled[i] -= overflow * (reducible[i] / reducible_total)
                scaled[i] = max(min_col_w, scaled[i])

    return scaled


def read_csv_table(path: Path, max_cols: Optional[int] = None) -> List[List[str]]:
    """Read a CSV table into a list of rows for ReportLab."""
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if max_cols is not None:
                row = row[:max_cols]
            rows.append([("" if v is None else str(v)) for v in row])
    return rows


def build_longtable(
    raw: List[List[str]],
    styles: Dict[str, ParagraphStyle],
    avail_width: float,
) -> LongTable:
    """Build a long table flowable for the PDF report."""
    if not raw:
        raw = [["(empty)"]]

    header = raw[0]
    body = raw[1:]

    ncols = max(1, len(header))

    # Choose font size by width/columns
    if ncols >= 10:
        fs = 7.5
    elif ncols >= 7:
        fs = 8.5
    else:
        fs = 9.0

    cell_style = ParagraphStyle("CellRuntime", parent=styles["TableCell"], fontSize=fs, leading=fs + 2)
    head_style = ParagraphStyle("HeadRuntime", parent=styles["TableHeader"], fontSize=fs, leading=fs + 2)

    def to_para(val: str, is_header: bool) -> Paragraph:
        """Perform the To para operation."""
        s = safe_escape(val)
        # light wrapping hints
        s = s.replace("://", "://<wbr/>").replace("/", "/<wbr/>")
        return Paragraph(s, head_style if is_header else cell_style)

    data: List[List[Any]] = []
    data.append([to_para(h, True) for h in header])
    for r in body:
        # pad
        rr = r + [""] * (ncols - len(r))
        data.append([to_para(v, False) for v in rr])

    col_widths = compute_col_widths(
        data=[[p.getPlainText() if isinstance(p, Paragraph) else str(p) for p in row] for row in raw[: min(len(raw), 60)]],
        avail_width=avail_width,
        font_name="Helvetica",
        font_size=fs,
        max_rows_probe=60,
    )

    t = LongTable(data, colWidths=col_widths, repeatRows=1)

    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("LINEBELOW", (0, 0), (-1, 0), 1, ACCENT),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    # zebra rows
    for i in range(1, len(data)):
        if i % 2 == 0:
            t.setStyle(TableStyle([("BACKGROUND", (0, i), (-1, i), LIGHT_BG)]))

    return t


# -----------------------------
# Artifact discovery and grouping
# -----------------------------

@dataclass(frozen=True)
class Artifact:
    name: str
    kind: str  # "figure" or "table"
    path: Path
    caption: str  # final caption text (non-empty)


def scan_output(output_dir: Path, manifest: Dict[str, ManifestItem]) -> List[Artifact]:
    """Scan the output directory and collect generated report artifacts."""
    artifacts: List[Artifact] = []

    tables_dir = output_dir / TABLES_DIRNAME
    figs_dir = output_dir / FIGURES_DIRNAME

    # Tables
    if tables_dir.exists():
        for p in sorted(tables_dir.glob("*.csv")):
            name = p.stem
            cap = caption_for(name, manifest, default_kind="table")
            artifacts.append(Artifact(name=name, kind="table", path=p, caption=cap))

    # Figures
    if figs_dir.exists():
        for p in sorted(figs_dir.iterdir()):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext in EMBED_FIG_EXTS or ext in EPS_EXTS:
                name = p.stem
                cap = caption_for(name, manifest, default_kind="figure")
                artifacts.append(Artifact(name=name, kind="figure", path=p, caption=cap))

    return artifacts


def caption_for(name: str, manifest: Dict[str, ManifestItem], default_kind: str) -> str:
    """
    Returns a useful, English caption.
    Priority:
      1) outputs_manifest.csv description (if non-empty and not nan-like)
      2) strong, pattern-based captions (never trivial "this is a plot of X")
      3) safe fallback based on filename (still informative)
    """

    def _var_from_stem(stem: str) -> str:
        """Perform the Var from stem operation."""
        s = stem.lower()
        s = s.replace("histogram", "").replace("barplot", "").replace("bar", "")
        s = s.replace("plot", "").replace("figure", "").replace("chart", "")
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    def _pretty(stem: str) -> str:
        """Perform the Pretty operation."""
        return humanize_name(stem)

    # 1) Manifest description if it is actually usable
    if name in manifest and not is_nan_like(manifest[name].description):
        desc = manifest[name].description.strip()
        # reject captions that are just filenames or useless placeholders
        if len(desc) >= 12 and (".png" not in desc.lower()) and (".csv" not in desc.lower()):
            return desc
        # otherwise keep going to generate something better

    n = name.lower()

    # ----------------------------
    # FIGURES: high-value captions
    # ----------------------------
    if default_kind == "figure":

        # Missingness
        if "missingness_heatmap" in n or ("missing" in n and "heatmap" in n):
            return (
                "Missingness matrix across rows and columns. This view helps detect structured "
                "missingness (block patterns) that may indicate systematic omissions in specific fields "
                "or subsets of the dataset."
            )
        if "missingness_barplot" in n or ("missing" in n and "bar" in n):
            return (
                "Per-column missing-value rate. Use this plot to identify fields with substantial "
                "missingness that may require explicit handling (e.g., exclusion, imputation, or reporting "
                "as a dataset limitation)."
            )

        # Labels
        if "label_distribution" in n:
            return (
                "Class balance across dataset labels. This figure supports reproducibility checks "
                "and contextualizes any supervised baselines or stratified splits derived from the label field."
            )

        # URLs
        if "url_count_histogram" in n or (n.startswith("url_") and "histogram" in n and "count" in n):
            return (
                "Distribution of the number of URLs per email. This figure reveals zero-inflation "
                "(emails with no links) and tail behavior (link-heavy messages), which are commonly informative "
                "in phishing campaigns."
            )
        if "top_url_domains" in n or ("url" in n and "domains" in n and "bar" in n):
            return (
                "Most frequent URL domains observed in the corpus. Concentration on a few domains may suggest "
                "reused infrastructure, whereas a flatter distribution suggests higher domain diversity."
            )

        # Attachments
        if "attachment_count_histogram" in n or ("attachment" in n and "count" in n and "histogram" in n):
            return (
                "Distribution of attachment counts per email. This plot separates messages without attachments "
                "from those carrying one or multiple files, which is relevant when characterizing payload delivery."
            )
        if "attachment_types" in n or ("attachment" in n and "types" in n and "bar" in n):
            return (
                "Attachment type / extension frequencies. This figure summarizes which file formats dominate "
                "in the dataset and supports qualitative assessment of likely payload styles."
            )
        if "attachment_unit_size_histogram" in n or ("attachment" in n and "size" in n and "histogram" in n):
            return (
                "Distribution of individual attachment sizes. This complements per-email totals by showing the "
                "single-file size profile, which can reveal typical payload magnitudes and extreme outliers."
            )

        # Length
        if "subject_length_histogram" in n:
            return (
                "Distribution of subject length (word-based). This plot highlights short, headline-like subjects "
                "versus longer subjects and helps identify skewness or rare extremes."
            )
        if "body_length_histogram" in n:
            return (
                "Distribution of body length (word-based). This figure helps characterize whether messages are "
                "brief prompts or long-form narratives, and whether length has heavy-tail behavior."
            )
        if "subject_char_length_histogram" in n:
            return (
                "Distribution of subject length (character-based). Character length is useful when subjects contain "
                "tokens, codes, or non-standard spacing that may not be captured well by word counts."
            )
        if "body_char_length_histogram" in n:
            return (
                "Distribution of body length (character-based). This plot complements word counts and can expose "
                "very long messages or templated content with repeated fragments."
            )

        # Hops / routing
        if "hops_count_histogram" in n or ("hops" in n and "histogram" in n):
            return (
                "Distribution of routing hop counts derived from header paths. Higher hop counts can indicate more "
                "complex routing or relaying behavior; the shape reveals whether most messages follow short or long paths."
            )

        # Temporal
        if "emails_over_time" in n or ("year_month" in n and ("emails" in n or "volume" in n)):
            return (
                "Email volume aggregated over time (e.g., year-month). This figure is used to inspect temporal coverage "
                "and potential bursts that may correspond to collection waves or concentrated campaigns."
            )

        # PoP / reliability
        if "pop_mean_intensity_by_principle" in n:
            return (
                "Average persuasion-intensity level per principle, computed from the stored annotation fields. "
                "This figure provides a cross-principle comparison of typical intensity levels under the adopted rating scale."
            )
        if "pop_icc_by_principle" in n or ("icc" in n and "principle" in n):
            return (
                "Inter-rater agreement (ICC) per persuasion principle. Higher values indicate stronger consistency "
                "across annotators, helping assess which principles are reliably perceived and scored."
            )

        # Label sanity
        if "label_sanity_structural_boxplots" in n:
            return (
                "Label-wise comparison of key structural variables (e.g., URL count, attachment count, and length measures). "
                "Separation between label distributions supports internal consistency; substantial overlap highlights regions "
                "where label boundaries may be less distinct and deserve manual review."
            )
        if "label_sanity_anomaly_scatter" in n:
            return (
                "Structural consistency view using URL count versus attachment count. Points flagged as anomalies "
                "indicate cases whose structural profile is unusual for their assigned label (e.g., phishing-labeled messages "
                "with no links/attachments, or benign-labeled messages with unexpectedly high link/payload presence)."
            )

        # Baselines
        if "baseline_confusion" in n:
            return (
                "Confusion matrix for a baseline classifier trained on text-derived features. This figure shows which "
                "classes are most frequently confused and provides a compact view of error structure beyond scalar metrics."
            )
        if "baseline_roc" in n:
            return (
                "ROC curve for a baseline classifier (when probabilistic scores are available). This figure shows the "
                "trade-off between true-positive and false-positive rates across thresholds, complementing accuracy/F1 summaries."
            )

        # Bias effect sizes
        if "bias_effect_sizes" in n and "vs_rest" in n:
            return (
                "Effect-size summary (standardized mean difference) comparing one label against all others across numeric features. "
                "Large absolute values indicate features strongly associated with that label, which is informative when assessing "
                "potential dataset bias or feature-label entanglement."
            )

        # Wordclouds
        if n.startswith("wordcloud"):
            return (
                "Word cloud summarizing prominent terms in the corresponding subset. This visualization is descriptive "
                "and intended to support qualitative inspection of vocabulary differences across subsets."
            )

        # Generic but still useful fallback for unknown figures
        if "histogram" in n:
            v = _pretty(_var_from_stem(n))
            return (
                f"Distribution summary for {v}. Use this figure to inspect concentration near zero, skewness, and "
                "the presence of rare extremes that may affect downstream modeling or summary statistics."
            )
        if "bar" in n:
            v = _pretty(_var_from_stem(n))
            return (
                f"Frequency summary for {v}. Use this figure to identify dominant categories and long-tail behavior, "
                "which is often relevant for corpus composition and reproducibility."
            )

        return (
            f"{_pretty(name)}. This figure is included as a descriptive artifact produced by the analysis pipeline; "
            "its filename indicates the corresponding variable subset and aggregation."
        )

    # ----------------------------
    # TABLES: high-value captions
    # ----------------------------
    if default_kind == "table":

        if "missingness_summary" in n:
            return (
                "Column-wise missing-value counts and percentages computed over the full dataset. "
                "This table supports transparent reporting of data completeness and highlights fields requiring caution."
            )
        if "label_distribution" in n:
            return (
                "Counts and proportions per label. This table documents class balance and is a required artifact "
                "for reproducibility of any stratified splits and supervised baselines."
            )
        if "url_statistics" in n:
            return (
                "Descriptive statistics for URL-related fields (e.g., per-email URL counts and derived aggregates). "
                "Use this table to complement histograms with exact quantiles and dispersion values."
            )
        if "url_domain_frequencies" in n:
            return (
                "Domain frequency table for extracted URLs. This table supports infrastructure characterization "
                "by listing domains and their observed counts."
            )
        if "url_scheme_frequencies" in n:
            return (
                "URL scheme frequencies (e.g., http/https). This table documents transport-layer patterns and "
                "can reveal outdated or unusual scheme usage in the corpus."
            )
        if "attachment_statistics" in n:
            return (
                "Per-email attachment statistics (counts and sizes). This table documents how attachments distribute "
                "across messages and supports comparison with figure-based summaries."
            )
        if "attachment_type_frequencies" in n:
            return (
                "Attachment type / extension frequency table. This table enumerates file-format prevalence to support "
                "payload characterization and reproducibility."
            )
        if "attachment_unit_size_statistics" in n:
            return (
                "Size distribution statistics computed at the single-attachment level. This table complements per-email totals "
                "and supports reporting of typical payload sizes and outliers."
            )
        if "email_length_statistics" in n:
            return (
                "Subject and body length descriptive statistics (word- and character-based). This table supports compact reporting "
                "of central tendency and spread for message length measures."
            )
        if "hops_statistics" in n:
            return (
                "Descriptive statistics for hop-count features derived from routing headers. This table documents typical routing complexity "
                "and complements the hop-count distribution figure."
            )
        if "emails_over_time" in n or ("year_month" in n and "emails" in n):
            return (
                "Time-binned email counts (e.g., year-month). This table provides exact counts underpinning the temporal volume figure "
                "and supports reproducible temporal filtering."
            )
        if "pop_descriptive_statistics_by_rater" in n:
            return (
                "Descriptive statistics for persuasion-intensity annotations stratified by principle and rater. "
                "This table documents score distributions and rater coverage to support reliability assessment."
            )
        if "pop_reliability_icc_and_correlations" in n:
            return (
                "Inter-rater reliability summary for persuasion principles, including ICC and pairwise rater correlations. "
                "This table quantifies annotation consistency and highlights principles with weaker agreement."
            )
        if "pop_label_association_tests" in n:
            return (
                "Non-parametric association tests between persuasion-intensity variables and labels. "
                "This table documents whether intensity distributions differ across label groups under the chosen tests."
            )
        if "baseline_metrics" in n:
            return (
                "Baseline model evaluation metrics computed on a held-out split. This table provides reproducible reference performance "
                "and complements confusion matrices and ROC curves."
            )
        if "label_sanity_check" in n:
            return (
                "Label-wise structural summary used for sanity checking. This table supports verification that structural patterns "
                "align with label assignments and highlights unexpected combinations for manual review."
            )
        if "label_sanity_anomalies" in n:
            return (
                "List of emails flagged as potential label anomalies using simple structural heuristics. This table is intended for "
                "manual inspection and possible relabeling, not as a definitive error list."
            )
        if "outputs_manifest" in n:
            return (
                "Inventory of all generated tables and figures under output/. This table is used to audit completeness and "
                "to ensure every artifact produced by the analysis pipeline is traceable."
            )
        if "dataset_schema" in n or "data_dictionary" in n:
            return (
                "Schema / data dictionary summary documenting column names, types, and related metadata. "
                "This table supports transparent dataset documentation and programmatic validation."
            )
        if n.startswith("bias_"):
            return (
                "Bias-diagnostic table derived from associations between labels and structural features or temporal bins. "
                "This table is included to make label-feature dependencies explicit for cautious downstream interpretation."
            )

        return (
            f"{_pretty(name)}. This table is an analysis artifact generated from the dataset; "
            "its filename indicates the corresponding variable subset and aggregation."
        )



def group_section(artifact_name: str) -> int:
    # Maps artifact names to one of the 15 sections in your menu.
    """Return the section group number for an artifact name."""
    n = artifact_name.lower()

    # 1 Integrated overview
    if "overview" in n or "data_summary" in n or n.startswith("data_"):
        return 1

    # 2 Missingness
    if "missing" in n:
        return 2

    # 3 Labels
    if n.startswith("label_"):
        return 3

    # 4 URLs
    if n.startswith("url_") or "urls" in n:
        return 4

    # 5 Attachments
    if n.startswith("attachment_") or n.startswith("attachments_"):
        return 5

    # 6 Email length
    if "length" in n or "char_len" in n or "word_len" in n:
        return 6

    # 7 Routing and hops
    if "hops" in n or "routing" in n:
        return 7

    # 8 Temporal
    if "year_month" in n or "over_time" in n or n.startswith("temporal_") or n.startswith("bias_temporal"):
        return 8

    # 9 Persuasion intensity and reliability
    if n.startswith("pop_") or "icc" in n or "intensity" in n:
        return 9

    # 10 Baseline
    if n.startswith("baseline_"):
        return 10

    # 11 Wordclouds
    if n.startswith("wordcloud"):
        return 11

    # 12 Examples
    if n.startswith("examples_") or "example" in n:
        return 12

    # 13 Bias
    if n.startswith("bias_"):
        return 13

    # 14 Schema
    if "schema" in n or "dictionary" in n:
        return 14

    # 15 Other
    return 15


SECTION_TITLES = {
    1: "Integrated EDA overview",
    2: "Data quality and missingness",
    3: "Label distribution",
    4: "URLs",
    5: "Attachments",
    6: "Email length",
    7: "Routing and hops",
    8: "Temporal distribution",
    9: "Persuasion intensity and reliability",
    10: "Baseline models",
    11: "Wordclouds",
    12: "Textual examples",
    13: "Potential bias analysis",
    14: "Dataset schema",
    15: "Other outputs",
}


SECTION_NARRATIVE = {
    1: [
        "This section provides an integrated snapshot of the dataset using compact, high-level visual summaries.",
        "It is intended to let the reader quickly verify overall scale and composition before inspecting topic-specific diagnostics in later sections.",
    ],
    2: [
        "This section summarizes missing-value diagnostics computed directly from the tabular fields.",
        "Figures highlight the missingness pattern and per-column missingness rates; the table reports counts and percentages per variable.",
    ],
    3: [
        "This section reports the distribution of labels in the corpus.",
        "Figures and tables provide absolute counts and relative proportions, which are essential for reproducibility and for understanding class balance.",
    ],
    4: [
        "This section summarizes URL-related structure extracted from the email content.",
        "It includes distributions of URL counts and any derived tabular summaries that describe URL presence and frequency patterns.",
    ],
    5: [
        "This section summarizes attachment-related structure at email level.",
        "Tables and figures describe counts, sizes, and type frequencies (when available), as computed from attachment metadata fields.",
    ],
    6: [
        "This section summarizes message-length statistics computed from subject and body fields.",
        "It includes character- and word-level length summaries and any derived distributions or comparisons by label or subset.",
    ],
    7: [
        "This section summarizes routing-related structure, such as hop counts derived from header routing paths.",
        "Tables typically report descriptive statistics and distributions that characterize transmission complexity per email.",
    ],
    8: [
        "This section summarizes temporal coverage based on the date field (when available).",
        "Artifacts provide counts aggregated by time bins (e.g., year-month), supporting inspection of collection density over time.",
    ],
    9: [
        "This section summarizes persuasion-intensity variables and any reliability statistics derived from multiple raters.",
        "Artifacts may include inter-rater agreement metrics and descriptive summaries by principle, rater, or label.",
    ],
    10: [
        "This section reports baseline model metrics computed from the dataset features.",
        "Tables provide evaluation results and are included to make the analysis reproducible from the stored outputs.",
    ],
    11: [
        "This section contains wordcloud artifacts generated from text fields or subsets.",
        "These visuals are descriptive summaries of prominent tokens under the corresponding filtering or grouping criteria.",
    ],
    12: [
        "This section provides tabular examples extracted from the corpus (e.g., sampled rows by label or high-intensity subsets).",
        "Examples are included for transparency and to allow manual inspection of representative cases.",
    ],
    13: [
        "This section reports potential bias diagnostics computed from structural features and labels.",
        "Artifacts include association tests and effect-size summaries intended to expose strong dependencies between labels and simple feature buckets.",
    ],
    14: [
        "This section exposes the dataset schema and data dictionary artifacts produced automatically from the stored fields.",
        "These tables document variable names, types, and derived metadata needed to reproduce downstream processing.",
    ],
    15: [
        "This section contains any remaining generated artifacts not categorized above.",
        "It also commonly includes the global outputs manifest, which inventories files under output/tables and output/figures.",
    ],
}


# -----------------------------
# PDF Assembly
# -----------------------------

def add_menu(story: List[Any], styles: Dict[str, ParagraphStyle]) -> None:
    """Add the navigation menu to the PDF story."""
    story.append(Paragraph('<a name="menu"/>', styles["Body"]))
    story.append(Paragraph("Menu", styles["H1"]))
    story.append(Paragraph(
        "Click a section to jump. You can also use the PDF outline/bookmarks panel.",
        styles["Body"],
    ))

    for sec_no in range(1, 16):
        title = SECTION_TITLES[sec_no]
        story.append(
            Paragraph(
                f'<link href="#sec_{sec_no}" color="{ACCENT}">{sec_no}. {safe_escape(title)}</link>',
                styles["MenuItem"],
            )
        )
    story.append(Spacer(1, 8))


def add_back_to_menu(story: List[Any], styles: Dict[str, ParagraphStyle]) -> None:
    """Add a back-to-menu link to the PDF story."""
    story.append(
        Paragraph(
            f'<link href="#menu" color="{ACCENT}">Back to menu</link>',
            styles["BackLink"],
        )
    )


def add_section_heading(
    story: List[Any],
    styles: Dict[str, ParagraphStyle],
    numbering: Numbering,
    sec_no: int,
    title: str,
) -> None:
    """Add the Heading section to the PDF report."""
    p = Paragraph(f'{sec_no}. {safe_escape(title)}', styles["H1"])
    # bookmark key used by DocTemplate.afterFlowable
    setattr(p, "_bookmark_key", f"sec_{sec_no}")
    story.append(Paragraph(f'<a name="sec_{sec_no}"/>', styles["Body"]))
    story.append(p)


def add_section_narrative(
    story: List[Any],
    styles: Dict[str, ParagraphStyle],
    sec_no: int,
) -> None:
    """Add the Narrative section to the PDF report."""
    for para in SECTION_NARRATIVE.get(sec_no, []):
        story.append(Paragraph(safe_escape(para), styles["Body"]))


def add_figure(
    story: List[Any],
    styles: Dict[str, ParagraphStyle],
    numbering: Numbering,
    fig_path: Path,
    fig_name: str,
    caption: str,
    avail_width: float,
    tmp_dir: Path,
) -> None:
    """Add a figure and its caption to the PDF story."""
    ext = fig_path.suffix.lower()

    # Convert EPS if needed
    emb_path = fig_path
    if ext in EPS_EXTS:
        png_path = tmp_dir / f"{fig_path.stem}.png"
        convert_eps_to_png(fig_path, png_path, dpi=300)
        emb_path = png_path

    if emb_path.suffix.lower() not in EMBED_FIG_EXTS:
        raise RuntimeError(f"Unsupported figure format for embedding: {emb_path}")

    # Scale image to available width, preserve aspect
    img = Image(str(emb_path))
    iw, ih = img.imageWidth, img.imageHeight
    if iw <= 0 or ih <= 0:
        raise RuntimeError(f"Invalid image dimensions: {emb_path}")

    scale = min(1.0, avail_width / float(iw))
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale

    fig_num = numbering.next_figure()
    story.append(img)

    cap_text = caption.strip()
    if is_nan_like(cap_text):
        cap_text = humanize_name(fig_name)

    story.append(
        Paragraph(
            f"<b>Figure {fig_num}.</b> {safe_escape(cap_text)}",
            styles["Caption"],
        )
    )


def add_table(
    story: List[Any],
    styles: Dict[str, ParagraphStyle],
    numbering: Numbering,
    csv_path: Path,
    table_name: str,
    caption: str,
    avail_width: float,
) -> None:
    """Add a table and its caption to the PDF story."""
    raw = read_csv_table(csv_path)

    tbl_num = numbering.next_table()
    cap_text = caption.strip()
    if is_nan_like(cap_text):
        cap_text = humanize_name(table_name)

    story.append(
        Paragraph(
            f"<b>Table {tbl_num}.</b> {safe_escape(cap_text)}",
            styles["Caption"],
        )
    )

    t = build_longtable(raw, styles=styles, avail_width=avail_width)
    story.append(t)
    story.append(Spacer(1, 6))


def build_pdf(output_dir: Path, pdf_path: Path) -> None:
    """Build the PDF report from the generated outputs."""
    styles = build_styles()
    manifest = load_manifest(output_dir)
    artifacts = scan_output(output_dir, manifest)

    # Group by section number
    by_section: Dict[int, List[Artifact]] = {i: [] for i in range(1, 16)}
    for a in artifacts:
        sec = group_section(a.name)
        by_section[sec].append(a)

    # Sort inside each section: figures first, then tables, alphabetical
    for sec in by_section:
        by_section[sec].sort(key=lambda x: (0 if x.kind == "figure" else 1, x.name))

    # Temporary directory for EPS->PNG conversions
    tmp_dir = output_dir / "_pdf_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Doc layout
    page_w, page_h = letter
    frame = Frame(
        DEFAULT_LEFT_RIGHT_MARGIN,
        DEFAULT_BOTTOM_MARGIN,
        page_w - 2 * DEFAULT_LEFT_RIGHT_MARGIN,
        page_h - DEFAULT_TOP_MARGIN - DEFAULT_BOTTOM_MARGIN,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        showBoundary=0,
    )

    def on_page(canvas, doc):
        """Perform the On page operation."""
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawRightString(page_w - DEFAULT_LEFT_RIGHT_MARGIN, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = SpaPhishDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=DEFAULT_LEFT_RIGHT_MARGIN,
        rightMargin=DEFAULT_LEFT_RIGHT_MARGIN,
        topMargin=DEFAULT_TOP_MARGIN,
        bottomMargin=DEFAULT_BOTTOM_MARGIN,
        title="SpaPhish Dataset - Primary PDF Report",
        author="",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=on_page)])

    story: List[Any] = []

    # Cover
    story.append(Paragraph("SpaPhish Dataset - Primary PDF Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Subtitle"]))
    story.append(Paragraph(
        "This report is built directly as a PDF (no HTML conversion). It contains internal navigation (menu links + outline) "
        "and embeds computed artifacts found under output/.",
        styles["Subtitle"],
    ))
    story.append(Spacer(1, 10))

    # TOC
    toc = TableOfContents()
    toc.levelStyles = [styles["TOCEntry"]]
    story.append(Paragraph("Table of contents", styles["TOCHeading"]))
    story.append(toc)
    story.append(PageBreak())

    # Menu
    add_menu(story, styles)
    story.append(PageBreak())

    # Sections
    numbering = Numbering()
    avail_width = page_w - 2 * DEFAULT_LEFT_RIGHT_MARGIN

    for sec_no in range(1, 16):
        numbering.new_section()
        title = SECTION_TITLES[sec_no]

        add_section_heading(story, styles, numbering, sec_no, title)
        add_section_narrative(story, styles, sec_no)

        sec_artifacts = by_section.get(sec_no, [])
        if not sec_artifacts:
            story.append(Paragraph("No artifacts were found for this section in the current output directory.", styles["Body"]))
            add_back_to_menu(story, styles)
            story.append(PageBreak())
            continue

        for a in sec_artifacts:
            if a.kind == "figure":
                add_figure(
                    story=story,
                    styles=styles,
                    numbering=numbering,
                    fig_path=a.path,
                    fig_name=a.name,
                    caption=a.caption,
                    avail_width=avail_width,
                    tmp_dir=tmp_dir,
                )
            elif a.kind == "table":
                add_table(
                    story=story,
                    styles=styles,
                    numbering=numbering,
                    csv_path=a.path,
                    table_name=a.name,
                    caption=a.caption,
                    avail_width=avail_width,
                )

        add_back_to_menu(story, styles)
        story.append(PageBreak())

    # Build
    doc.build(story)

    # Cleanup temp dir (optional)
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass


# -----------------------------
# CLI
# -----------------------------

def main() -> int:
    # Default output-dir is resolved relative to the project root (one level up).
    """Run the PDF report generator and return an exit code."""
    default_output = str(Path(__file__).resolve().parent.parent / "output")

    ap = argparse.ArgumentParser(description="Generate SpaPhish PDF report.")
    ap.add_argument(
        "--output-dir",
        type=str,
        default=default_output,
        help="Directory containing tables/ and figures/ (default: <project_root>/output)",
    )
    ap.add_argument(
        "--pdf",
        type=str,
        default="SpaPhish_Report.pdf",
        help="Output PDF file path (default: SpaPhish_Report.pdf in the current directory)",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    pdf_path = Path(args.pdf).resolve()

    if not output_dir.exists():
        raise SystemExit(f"output-dir not found: {output_dir}")

    build_pdf(output_dir=output_dir, pdf_path=pdf_path)
    print(f"OK: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
