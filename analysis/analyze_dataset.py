#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze_dataset.py

Generates Data in Brief–ready descriptive statistics, tables and figures
from the SpaPhish dataset.

Usage:
    python analysis/analyze_dataset.py

Assumptions:
- This script lives in the analysis/ subdirectory of the project root.
- The dataset is located at: <project_root>/data/Spaphish dataset - DiB.csv
  (an Excel .xlsx version is also accepted)
- All outputs are written to: <project_root>/output/
    <project_root>/output/tables/   -> CSV tables
    <project_root>/output/figures/  -> PNG and EPS figures
"""

from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

import re
import html
import json
from collections import Counter
from scipy.stats import kruskal, chi2_contingency, mannwhitneyu

from wordcloud import WordCloud


# =====================================================================
# Global visual style (journal-like, clean, Data in Brief–friendly)
# =====================================================================

plt.rcParams.update(
    {
        # Default figure size and resolution
        "figure.figsize": (6, 4),
        "figure.dpi": 300,
        # Consistent font family
        "font.family": "DejaVu Sans",
        # Titles and labels
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        # Grid
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        # Figure edges
        "savefig.bbox": "tight",
    }
)

BAR_COLOR = "#2F6690"
HIST_COLOR = "#2F6690"
LINE_COLOR = "#2F6690"
BOX_FACE_COLOR = "#A9C5E8"
BOX_EDGE_COLOR = "#1B3A57"


def _format_axes(ax, title: str, xlabel: str, ylabel: str):
    """Apply consistent formatting to axes."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


# Correct canonical names (the old dataset had a typo: "commitement_*")
PRINCIPLES = [
    "authority",
    "social_proof",
    "liking_similarity_deception",
    "commitment_integrity_reciprocation",
    "distraction",
]

# Aliases for backward compatibility with older dataset versions
PRINCIPLE_ALIASES = {
    "commitement_integrity_reciprocation": "commitment_integrity_reciprocation",
}

# Human-readable descriptions for key artifacts (used in the outputs manifest)
OUTPUT_DESCRIPTIONS = {
    # Overview
    "data_summary_overview": "Multi-panel overview figure of the SpaPhish dataset",
    # Label
    "label_distribution": "Table with raw label codes, semantic names and frequencies",
    "label_distribution_bar": "Bar plot of email counts per label",
    # URLs
    "url_statistics": "Descriptive statistics for url_count per email",
    "url_domain_frequencies": "Frequencies of URL domains",
    "url_scheme_frequencies": "Frequencies of URL schemes (http/https/…)",
    "url_count_histogram": "Histogram of URL count per email",
    # Attachments
    "attachment_statistics": "Descriptive statistics for attachments_count and attachments_total_size at email level",
    "attachment_type_frequencies": "Frequencies of attachment types/extensions",
    "attachment_count_histogram": "Histogram of attachment count per email",
    "attachment_types_bar": "Bar plot of most frequent attachment types/extensions",
    "attachment_unit_size_statistics": "Descriptive statistics for individual attachment sizes",
    "attachment_unit_size_histogram": "Histogram of individual attachment sizes",
    # Hops
    "hops_statistics": "Descriptive statistics for hops_count per email",
    "hops_count_histogram": "Histogram of hops_count per email",
    # Temporal
    "emails_over_time_year_month": "Email volume over time (year-month)",
    # PoP descriptive stats and reliability
    "pop_descriptive_statistics_by_rater": "Descriptive statistics of PoP intensity by principle and annotator",
    "pop_reliability_icc_and_correlations": "Inter-rater reliability (ICC and Spearman) for PoP principles",
    "pop_mean_intensity_by_principle": "Mean PoP intensity per principle",
    "pop_icc_by_principle": "Bar plot of ICC(2,1) per PoP principle",
    # PoP vs Label
    "pop_label_association_tests": "Non-parametric tests (Mann-Whitney/Kruskal) for PoP intensity vs Label",
    # PoP justifications
    "pop_justifications_overview": "Coverage and basic length stats of textual justifications (justif_*)",
    # Label sanity
    "label_sanity_check": "Structural sanity-check statistics by label",
    "label_sanity_anomalies": "List of structurally suspicious label assignments",
    "label_sanity_structural_boxplots": "Boxplots of structural variables by label",
    "label_sanity_anomaly_scatter": "Scatter plot of url_count vs attachments_count highlighting anomalies",
}

TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NONWORD_RE = re.compile(r"[^a-záéíóúüñ0-9 ]", flags=re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")

def _clean_text_for_wordcloud(text: str) -> str:
    """Normalize text so it can be rendered reliably in a word cloud."""
    if not isinstance(text, str):
        return ""

    # Unescape HTML
    text = html.unescape(text)

    # Remove HTML tags
    text = TAG_RE.sub(" ", text)

    # Remove URLs
    text = URL_RE.sub(" ", text)

    # Remove emails
    text = EMAIL_RE.sub(" ", text)

    # Remove obvious HTML/CSS garbage words
    text = re.sub(
        r"(width|height|style|color|font|align|border|table|img|span|div)\w*",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Keep only letters/numbers/basic spaces (lowercased)
    text = NONWORD_RE.sub(" ", text.lower())

    # Collapse whitespace
    text = MULTISPACE_RE.sub(" ", text)

    return text.strip()


# =============================================================================
# Utility helpers
# =============================================================================
def _normalize_label_semantic_series(label_series: pd.Series) -> pd.Series:
    """
    Normalise the Label column to coherent semantic values.

    Rules:
      - Binary numeric (e.g. {0, 1}):
          lower value -> 'legitimate'
          higher value -> 'phishing'
      - Text or mixed: common variants are mapped to 'phishing' or
        'legitimate'; everything else is lowercased as-is.
    """
    s = label_series.copy()
    sem = []

    # Case 1: binary numeric label
    if pd.api.types.is_numeric_dtype(s):
        uniq = sorted([v for v in s.dropna().unique()])
        if len(uniq) == 2:
            neg, pos = uniq[0], uniq[1]
            for v in s:
                if pd.isna(v):
                    sem.append(np.nan)
                elif v == pos:
                    sem.append("phishing")
                elif v == neg:
                    sem.append("legitimate")
                else:
                    sem.append(str(v).strip().lower())
            return pd.Series(sem, index=s.index)

    # Case 2: text / mixed label
    for v in s:
        if pd.isna(v):
            sem.append(np.nan)
            continue
        t = str(v).strip().lower()
        if t in ("1", "phishing", "spam", "attack", "malicious"):
            sem.append("phishing")
        elif t in ("0", "legitimate", "legit", "benign", "ham", "legitimate"):
            sem.append("legitimate")
        else:
            sem.append(t)
    return pd.Series(sem, index=s.index)


def _build_label_semantic_mapping(label_series: pd.Series) -> dict:
    """
    Return a dict {raw_value -> semantic_label} using
    _normalize_label_semantic_series.
    """
    sem = _normalize_label_semantic_series(label_series)
    mapping: dict = {}
    for raw, sm in zip(label_series, sem):
        if pd.isna(raw) or pd.isna(sm):
            continue
        if raw not in mapping:
            mapping[raw] = sm
    return mapping


def resolve_principle_column(df: pd.DataFrame, principle: str) -> str | None:
    """
    Return the correct column name for a persuasion principle.

    - Returns the canonical name if it exists in the DataFrame.
    - Falls back to legacy aliases (e.g. the 'commitement_*' typo).
    - Returns None if no matching column is found.
    """
    if principle in df.columns:
        return principle

    # Check legacy aliases (older dataset versions)
    for old_name, canonical in PRINCIPLE_ALIASES.items():
        if canonical == principle and old_name in df.columns:
            return old_name

    return None


def _sanitize_token_for_filename(text: str) -> str:
    """Compatibility wrapper: sanitize a value for safe use in filenames."""
    return _sanitize_for_filename(str(text))


def _build_text_corpus(df: pd.DataFrame, mask=None) -> str:
    """
    Build a text corpus by concatenating subject and body from selected rows.

    Parameters
    ----------
    df : DataFrame containing at least 'subject' and 'body' columns.
    mask : optional boolean Series to filter rows before concatenation.
    """
    if mask is not None:
        df = df[mask]

    if "subject" not in df.columns or "body" not in df.columns:
        return ""

    subj = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    texts = subj + " " + body
    big = " ".join(texts.tolist())
    big = big.replace("\n", " ")
    return big


def _make_wordcloud(text: str, out_stem: Path, title: str | None = None):
    """
    Generate a word cloud and save it as both PNG and EPS.

    - Cleans HTML, URLs, and other noise from the text.
    - Filters Spanish stopwords.
    - Dynamically removes hyper-generic words (e.g. 'correo', 'datos') and
      tokens shorter than 3 characters that would dominate the cloud.

    Parameters
    ----------
    text : raw text to visualise.
    out_stem : output path without extension (both .png and .eps are created).
    title : optional title displayed above the word cloud.
    """
    if WordCloud is None:
        print("[WARN] wordcloud library not available; skipping wordcloud generation.")
        return

    cleaned = _clean_text_for_wordcloud(text)
    if not cleaned:
        print(f"[WARN] Empty text for wordcloud {out_stem}, skipping.")
        return

    # Basic Spanish stopwords
    spanish_stopwords = {
        "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas",
        "y", "o", "u", "que", "como", "con", "sin", "por", "para", "sobre",
        "entre", "hasta", "desde", "donde", "cuando",
        "a", "al", "se", "su", "sus", "tu", "tus", "mi", "mis", "nuestro",
        "nuestra", "nuestros", "nuestras",
        "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
        "aqui", "aquí", "alli", "allí", "ahi", "ahí", "ya", "hoy", "manana", "mañana",
        "lo", "le", "les", "nos", "te",
        "es", "son", "fue", "han", "ha", "hay", "ser", "estar", "esta", "está", "estan", "están",
        "suyo", "suya", "suyos", "suyas",
        "mas", "más", "menos", "muy", "tambien", "también", "solo", "sólo", "cada",
        "usted", "ustedes",
        # Common email / URL noise tokens
        "http", "https", "www", "com", "mx", "org", "net",
        "mailto", "html",
        # Very short tokens that rarely add value
        "q", "d", "s", "t",
    }

    # Simple tokenisation
    tokens = cleaned.split()

    # Remove stopwords and short tokens
    tokens = [
        w
        for w in tokens
        if w not in spanish_stopwords and len(w) >= 3
    ]

    if not tokens:
        print(f"[WARN] All tokens removed by stopword/length filters for {out_stem}.")
        return

    # Dynamic filter: remove the most frequent tokens if they are too generic
    freq = Counter(tokens)

    generic_super_common = {
        # Ultra-generic email-domain words
        "correo", "email", "mail", "mensaje", "mensajes",
        "informacion", "información", "info",
        "dato", "datos",
        "sistema", "servicio", "cuenta", "usuario", "usuarios",
        "centro", "universidad", "instituto",
        "google",
    }

    dynamic_stop = set()
    for w, c in freq.most_common(40):  # inspect top 40 tokens
        if (
            w in generic_super_common
            or len(w) <= 3
        ):
            dynamic_stop.add(w)

    # Apply the dynamic filter
    filtered_tokens = [w for w in tokens if w not in dynamic_stop]

    # If the dynamic filter is too aggressive, fall back to the basic filter only
    if len(filtered_tokens) < 20:
        filtered_tokens = tokens

    text_for_wc = " ".join(filtered_tokens)
    if not text_for_wc.strip():
        print(f"[WARN] No text left for wordcloud {out_stem} after filtering.")
        return

    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        max_words=200,
        collocations=False,
        stopwords=spanish_stopwords | dynamic_stop,
    ).generate(text_for_wc)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout()

    png_path = out_stem.with_suffix(".png")
    eps_path = out_stem.with_suffix(".eps")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(eps_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved wordcloud: {png_path} and {eps_path}")


def generate_wordclouds(df: pd.DataFrame, figs_dir: Path):
    """
    Generate word clouds from the subject and body text fields.

    Produces:
      - wordcloud_global.(png|eps)          — full corpus
      - wordcloud_label_<label>.*           — one per class label
      - wordcloud_<principle>_label_<label>.* — PoP-filtered per label
    """
    # Global word cloud
    print("[INFO] Generating global wordcloud...")
    global_text = _build_text_corpus(df)
    _make_wordcloud(
        global_text, figs_dir / "wordcloud_global", "Global wordcloud (subject + body)"
    )

    # Per label
    if "Label" in df.columns:
        labels = sorted(df["Label"].dropna().unique(), key=lambda x: str(x))
        for lab in labels:
            mask = df["Label"] == lab
            text = _build_text_corpus(df, mask)
            token = _sanitize_token_for_filename(lab)
            stem = figs_dir / f"wordcloud_label_{token}"
            title = f"Wordcloud for Label = {lab}"
            print(f"[INFO] Generating wordcloud for Label={lab}...")
            _make_wordcloud(text, stem, title)
    else:
        print("[WARN] Column 'Label' not found; skipping per-label wordclouds.")

    # Per persuasion principle + label (only emails with intensity > 0)
    if "Label" in df.columns:
        labels = sorted(df["Label"].dropna().unique(), key=lambda x: str(x))
        for principle in PRINCIPLES:
            if principle not in df.columns:
                print(
                    f"[WARN] Column '{principle}' not found; skipping wordclouds for this principle."
                )
                continue

            intensity = pd.to_numeric(df[principle], errors="coerce")
            for lab in labels:
                mask = (df["Label"] == lab) & (intensity > 0)
                n = mask.sum()
                if n < 10:
                    # Too few observations to produce a meaningful word cloud
                    continue
                text = _build_text_corpus(df, mask)
                lab_token = _sanitize_token_for_filename(lab)
                stem = figs_dir / f"wordcloud_{principle}_label_{lab_token}"
                title = f"{principle} wording, Label={lab}"
                print(
                    f"[INFO] Generating wordcloud for {principle}, Label={lab} (n={n})..."
                )
                _make_wordcloud(text, stem, title)
    else:
        print("[WARN] Column 'Label' not found; skipping PoP+Label wordclouds.")


def generate_data_summary_figure(df: pd.DataFrame, figs_dir: Path):
    """
    Create a six-panel overview figure of the dataset.

      Panel 1: Label distribution (phishing / legitimate)
      Panel 2: URL count per email
      Panel 3: Attachment count per email
      Panel 4: Body length (words)
      Panel 5: Email volume over time (year-month)
      Panel 6: Mean PoP intensity by principle

    Saves: data_summary_overview.(png|eps) in figs_dir.
    """
    # Work on a copy with length features pre-computed
    df_work = _add_length_features(df.copy())

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes = axes.ravel()

    # -------------------------------------------------------------------------
    # Panel 1: Label distribution
    # -------------------------------------------------------------------------
    ax = axes[0]
    if "Label" in df_work.columns:
        label_series = df_work["Label"]
        counts = label_series.value_counts(dropna=False)
        total = counts.sum()

        # Map raw label codes to semantic names (phishing / legitimate)
        mapping = _build_label_semantic_mapping(label_series)
        label_codes = list(counts.index)
        label_names = [
            mapping.get(val, str(val)) if not pd.isna(val) else "NaN"
            for val in label_codes
        ]
        values = counts.values

        ax.bar(label_names, values, color=BAR_COLOR, edgecolor="black")
        ax.set_title("Label distribution")
        ax.set_xlabel("Label")
        ax.set_ylabel("Number of emails")
        ax.tick_params(axis="x", rotation=45)

        # The table is generated by analyze_label_distribution; here only the figure
    else:
        ax.text(0.5, 0.5, "No 'Label' column", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------------------------------------------------------
    # Panel 2: URL count per email
    # -------------------------------------------------------------------------
    ax = axes[1]
    if "url_count" in df_work.columns:
        url_series = _safe_numeric_series(df_work, "url_count")
        if not url_series.dropna().empty:
            ax.hist(url_series.dropna(), bins=30, color=HIST_COLOR, edgecolor="black")
            _format_axes(ax, "URL count per email", "Number of URLs", "Frequency")
        else:
            ax.text(0.5, 0.5, "No valid url_count", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "No 'url_count' column", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------------------------------------------------------
    # Panel 3: Attachment count per email
    # -------------------------------------------------------------------------
    ax = axes[2]
    if "attachments_count" in df_work.columns:
        att_series = _safe_numeric_series(df_work, "attachments_count")
        if not att_series.dropna().empty:
            ax.hist(att_series.dropna(), bins=30, color=HIST_COLOR, edgecolor="black")
            _format_axes(
                ax, "Attachment count per email", "Number of attachments", "Frequency"
            )
        else:
            ax.text(0.5, 0.5, "No valid attachments_count", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "No 'attachments_count' column", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------------------------------------------------------
    # Panel 4: Body length (words)
    # -------------------------------------------------------------------------
    ax = axes[3]
    if "body_word_len" in df_work.columns:
        body_words = _safe_numeric_series(df_work, "body_word_len")
        if not body_words.dropna().empty:
            ax.hist(body_words.dropna(), bins=40, color=HIST_COLOR, edgecolor="black")
            _format_axes(
                ax, "Body length (words)", "Number of words in body", "Frequency"
            )
        else:
            ax.text(0.5, 0.5, "No valid body_word_len", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "No 'body_word_len' column", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------------------------------------------------------
    # Panel 5: Email volume over time (year-month)
    # -------------------------------------------------------------------------
    ax = axes[4]
    df_parsed = _parse_dates(df_work)
    if "date_parsed" in df_parsed.columns and df_parsed["date_parsed"].notna().any():
        valid_dates = df_parsed["date_parsed"].dropna()
        period = valid_dates.dt.to_period("M")
        counts = period.value_counts().sort_index()

        if not counts.empty:
            x_vals = counts.index.astype(str)
            ax.plot(
                x_vals,
                counts.values,
                marker="o",
                linestyle="-",
                color=LINE_COLOR,
            )
            ax.set_title("Email volume over time (Y-M)")
            ax.set_xlabel("Year-Month")
            ax.set_ylabel("Number of emails")
            # Reduce number of x-axis ticks if the range is wide
            if len(x_vals) > 10:
                step = max(1, int(np.ceil(len(x_vals) / 10)))
                idxs = np.arange(0, len(x_vals), step)
                ax.set_xticks(idxs)
                ax.set_xticklabels([x_vals[i] for i in idxs], rotation=45, ha="right")
            else:
                ax.tick_params(axis="x", rotation=45)
        else:
            ax.text(0.5, 0.5, "No temporal variation", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "No valid dates", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------------------------------------------------------
    # Panel 6: Mean PoP intensity by principle
    # -------------------------------------------------------------------------
    ax = axes[5]
    mean_by_principle = []

    for principle in PRINCIPLES:
        # Use aggregated column if available (or resolve legacy alias)
        col_resolved = resolve_principle_column(df_work, principle)
        if col_resolved is not None:
            s = pd.to_numeric(df_work[col_resolved], errors="coerce")
        else:
            # Fallback: raters A/B/C
            raters = [
                f"{principle}_A",
                f"{principle}_B",
                f"{principle}_C",
            ]
            existing_raters = [c for c in raters if c in df_work.columns]
            if not existing_raters:
                continue
            s = pd.concat(
                [pd.to_numeric(df_work[c], errors="coerce") for c in existing_raters],
                axis=1,
            ).mean(axis=1)

        s = s.dropna()
        if s.empty:
            continue

        mean_by_principle.append((principle, float(s.mean())))

    if mean_by_principle:
        principles_names, means = zip(*mean_by_principle)
        ax.bar(principles_names, means, color=BAR_COLOR, edgecolor="black")
        ax.set_title("Mean PoP intensity by principle")
        ax.set_xlabel("Principle")
        ax.set_ylabel("Mean intensity")

        # Rotate and align x-axis tick labels
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    else:
        ax.text(0.5, 0.5, "No PoP columns found", ha="center", va="center")
        ax.set_axis_off()

    plt.tight_layout()
    save_figure(fig, figs_dir, "data_summary_overview")


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load dataset from Excel or CSV depending on file extension."""
    suffix = input_path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    elif suffix in [".csv", ".txt"]:
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")
    return df


def ensure_output_dirs(base_output: Path) -> dict:
    """
    Create base, tables, and figures directories if they do not exist.

    Structure:
      base_output/
        tables/
        figures/
    """
    base_output.mkdir(parents=True, exist_ok=True)
    tables_dir = base_output / "tables"
    figs_dir = base_output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    return {"tables": tables_dir, "figures": figs_dir}


def save_table(df: pd.DataFrame, tables_dir: Path, filename: str):
    """Save a DataFrame as CSV in the tables directory."""
    output_path = tables_dir / filename
    df_out = df
    # If the index is named or non-default, promote it to a column to avoid "Unnamed" columns
    if (
        not isinstance(df.index, pd.RangeIndex)
        or df.index.name is not None
        or any(df.index.names)
    ):
        df_out = df.reset_index()
    df_out.to_csv(output_path, index=False)
    print(f"[INFO] Saved table: {output_path}")


def save_figure(fig: plt.Figure, figs_dir: Path, basename: str):
    """
    Save a figure as both PNG and EPS with consistent settings.

    Parameters
    ----------
    basename : base file name without extension; two files are created:
        <basename>.png
        <basename>.eps
    """
    png_path = figs_dir / f"{basename}.png"
    eps_path = figs_dir / f"{basename}.eps"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved figure: {png_path}")
    print(f"[INFO] Saved figure: {eps_path}")


def generate_outputs_manifest(tables_dir: Path, figs_dir: Path):
    """
    Generate a manifest of all output artefacts (tables and figures).

    Creates outputs_manifest.csv in tables_dir with columns:
      - name        : base stem without extension
      - path_rel    : path relative to the output/ directory
      - kind        : 'table' or 'figure'
      - format      : file extension ('csv', 'png', 'eps', …)
      - description : human-readable description from OUTPUT_DESCRIPTIONS (empty if unknown)
    """
    records = []

    # CSV tables in tables_dir
    for f in sorted(tables_dir.glob("*.csv")):
        stem = f.stem
        records.append(
            {
                "name": stem,
                "path_rel": str(f.relative_to(tables_dir.parent)),
                "kind": "table",
                "format": "csv",
                "description": OUTPUT_DESCRIPTIONS.get(stem, ""),
            }
        )

    # PNG and EPS figures in figs_dir
    for ext in ("png", "eps"):
        for f in sorted(figs_dir.glob(f"*.{ext}")):
            stem = f.stem
            records.append(
                {
                    "name": stem,
                    "path_rel": str(f.relative_to(figs_dir.parent)),
                    "kind": "figure",
                    "format": ext,
                    "description": OUTPUT_DESCRIPTIONS.get(stem, ""),
                }
            )

    if not records:
        print("[WARN] No outputs found to build manifest.")
        return

    manifest_df = pd.DataFrame.from_records(records)
    manifest_path = tables_dir / "outputs_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[INFO] Saved outputs manifest: {manifest_path}")


def _safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric Series, coercing errors and ignoring missing."""
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataset.")
    return pd.to_numeric(df[col], errors="coerce")


def _add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add character and word length features for subject and body."""
    df = df.copy()
    df["subject_str"] = df["subject"].astype(str)
    df["body_str"] = df["body"].astype(str)

    df["subject_char_len"] = df["subject_str"].str.len()
    df["body_char_len"] = df["body_str"].str.len()

    df["subject_word_len"] = df["subject_str"].str.split().str.len()
    df["body_word_len"] = df["body_str"].str.split().str.len()

    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 'date' column to datetime, if present."""
    df = df.copy()
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date_parsed"] = pd.NaT
    return df


def _parse_urls_field(urls_value: str):
    """
    Parse a urls cell value into a list of (domain, scheme) tuples.

    Strict policy:
    - Only tokens that start with 'http://' or 'https://' are considered.
    - Brackets and quotes typical of list-like representations are stripped.
    - Anything else (e.g. 'format=auto', 'width=650') is ignored.
    """
    if pd.isna(urls_value):
        return []

    text = str(urls_value).strip()
    if not text:
        return []

    # Strip brackets and quotes from list-like representations
    for ch in ["[", "]", "(", ")", "'", '"']:
        text = text.replace(ch, " ")

    # Normalise separators
    for sep in [";", "|"]:
        text = text.replace(sep, " ")

    # Split by comma first, then by whitespace
    raw_tokens = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        raw_tokens.extend(part.split())

    tokens = []
    for tok in raw_tokens:
        tok = tok.strip()
        if not tok:
            continue
        # Keep only explicit URLs (with scheme)
        if tok.startswith("http://") or tok.startswith("https://"):
            tokens.append(tok)

    result = []
    for tok in tokens:
        try:
            parsed = urlparse(tok)
        except ValueError:
            # Completely invalid URL — skip it
            continue

        domain = parsed.netloc.lower().strip()
        scheme = parsed.scheme.lower().strip()

        if not domain:
            continue

        if domain.startswith("www."):
            domain = domain[4:]

        result.append((domain, scheme or ""))

    return result


def _sanitize_for_filename(text: str) -> str:
    """Sanitize a string to be safely used in filenames."""
    text = str(text).lower()
    allowed = []
    for ch in text:
        if ch.isalnum() or ch in "-_":
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed)


def _parse_attachment_types_field(types_value: str):
    """Parse an attachments_types cell into a list of cleaned types/extensions."""
    if pd.isna(types_value):
        return []

    text = str(types_value).strip()
    if not text:
        return []

    for ch in ["[", "]", "(", ")", "'", '"']:
        text = text.replace(ch, " ")

    for sep in [";", "|"]:
        text = text.replace(sep, " ")

    parts = text.split()
    tokens = []
    for p in parts:
        for sub in p.split(","):
            sub = sub.strip().lower()
            if sub:
                tokens.append(sub)

    cleaned = []
    for t in tokens:
        if t.startswith("."):
            t = t[1:]
        cleaned.append(t)

    return cleaned


# =============================================================================
# ICC computation (two-way random effects, single measures, absolute agreement)
# =============================================================================


def compute_icc_2_1(data: np.ndarray) -> float:
    """
    Compute ICC(2,1) for a 2D array (subjects x raters).

    Returns NaN if computation is not possible.
    """
    if data.ndim != 2:
        raise ValueError("Data for ICC must be a 2D array (subjects x raters).")

    n, k = data.shape
    if n < 2 or k < 2:
        return np.nan

    mask = ~np.isnan(data).any(axis=1)
    data = data[mask]
    n, k = data.shape
    if n < 2:
        return np.nan

    mean_per_subject = data.mean(axis=1, keepdims=True)
    mean_per_rater = data.mean(axis=0, keepdims=True)
    grand_mean = data.mean()

    ss_between_subjects = k * np.sum((mean_per_subject - grand_mean) ** 2)
    ss_between_raters = n * np.sum((mean_per_rater - grand_mean) ** 2)
    ss_residual = np.sum((data - mean_per_subject - mean_per_rater + grand_mean) ** 2)

    df_between_subjects = n - 1
    df_between_raters = k - 1
    df_residual = (n - 1) * (k - 1)

    ms_between_subjects = ss_between_subjects / df_between_subjects
    ms_between_raters = (
        ss_between_raters / df_between_raters if df_between_raters > 0 else 0.0
    )
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0.0

    denominator = (
        ms_between_subjects
        + (k - 1) * ms_residual
        + (k * (ms_between_raters - ms_residual) / n)
    )
    if denominator == 0:
        return np.nan

    icc = (ms_between_subjects - ms_residual) / denominator
    return float(icc)


# =============================================================================
# Metric computations and figure generation
# =============================================================================


def analyze_label_distribution(df: pd.DataFrame, tables_dir: Path, figs_dir: Path):
    """
    Compute label frequency counts and produce a bar chart.

    Outputs:
        - label_distribution.csv
        - label_distribution_bar.(png|eps)
    """
    if "Label" not in df.columns:
        raise KeyError("Column 'Label' not found in dataset.")

    label_series = df["Label"]

    # Raw value counts
    label_counts = label_series.value_counts(dropna=False)
    label_percent = label_counts / label_counts.sum() * 100.0

    # Map raw codes to semantic names (phishing / legitimate / other)
    mapping = _build_label_semantic_mapping(label_series)

    label_codes = list(label_counts.index)
    label_names = [
        mapping.get(val, str(val)) if not pd.isna(val) else "NaN" for val in label_codes
    ]

    # Table: include both the raw code and the semantic name
    label_df = pd.DataFrame(
        {
            "label_code": label_codes,
            "label_name": label_names,
            "count": label_counts.values,
            "percentage": label_percent.values,
        }
    )

    save_table(label_df, tables_dir, "label_distribution.csv")

    # Figure: X-axis uses semantic label names
    fig, ax = plt.subplots()
    ax.bar(
        label_names,
        label_counts.values,
        color=BAR_COLOR,
        edgecolor="black",
    )
    _format_axes(ax, "Label Distribution", "Label", "Number of emails")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_figure(fig, figs_dir, "label_distribution_bar")


def analyze_urls(df: pd.DataFrame, tables_dir: Path, figs_dir: Path):
    """
    Analyse URL-related fields and produce statistics and figures.

    Outputs:
        - url_statistics.csv
        - url_count_histogram.(png|eps)
        - url_domain_frequencies.csv  (if 'urls' column exists)
        - top_url_domains_bar.(png|eps)
        - url_scheme_frequencies.csv
    """
    if "url_count" not in df.columns:
        print("[WARN] Column 'url_count' not found; skipping URL analysis.")
        return

    url_series = _safe_numeric_series(df, "url_count")

    # -------------------------
    # Descriptive statistics table
    # -------------------------
    desc = url_series.describe(percentiles=[0.25, 0.5, 0.75])

    rows = []
    # Global statistics (all rows)
    for stat_name, value in desc.items():
        rows.append(
            {
                "stat": str(stat_name),
                "url_count": float(value),
            }
        )

    # Additional statistics for rows with at least one URL
    non_zero = url_series[url_series > 0]
    if not non_zero.empty:
        nz_stats = {
            "non_zero_count": int(non_zero.count()),
            "non_zero_mean": float(non_zero.mean()),
            "non_zero_max": float(non_zero.max()),
        }
        for stat_name, value in nz_stats.items():
            rows.append(
                {
                    "stat": str(stat_name),
                    "url_count": float(value),
                }
            )

    url_stats = pd.DataFrame.from_records(rows)
    save_table(url_stats, tables_dir, "url_statistics.csv")

    # -------------------------
    # Histogram
    # -------------------------
    fig, ax = plt.subplots()
    ax.hist(url_series.dropna(), bins=30, color=HIST_COLOR, edgecolor="black")
    _format_axes(ax, "URL Count per Email", "Number of URLs", "Frequency")
    plt.tight_layout()
    save_figure(fig, figs_dir, "url_count_histogram")
    print(f"[INFO] Saved figure: {figs_dir}")

    # -------------------------
    # URL domains and schemes
    # -------------------------
    if "urls" in df.columns:
        from collections import Counter

        domain_counter = Counter()
        scheme_counter = Counter()

        for val in df["urls"]:
            pairs = _parse_urls_field(val)
            for dom, sch in pairs:
                domain_counter[dom] += 1
                if sch:
                    scheme_counter[sch] += 1

        if domain_counter:
            total = sum(domain_counter.values())
            dom_df = pd.DataFrame.from_records(
                [
                    {"domain": d, "count": c, "percentage": (c / total) * 100.0}
                    for d, c in domain_counter.most_common()
                ]
            ).set_index("domain")
            save_table(dom_df, tables_dir, "url_domain_frequencies.csv")

            top_n = min(20, len(dom_df))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(
                dom_df.iloc[:top_n].index,
                dom_df.iloc[:top_n]["count"],
                color=BAR_COLOR,
                edgecolor="black",
            )
            _format_axes(ax, "Top URL Domains", "Domain", "Number of occurrences")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            save_figure(fig, figs_dir, "top_url_domains_bar")
            print(f"[INFO] Saved figure: {figs_dir}")

        if scheme_counter:
            total_s = sum(scheme_counter.values())
            sch_df = pd.DataFrame.from_records(
                [
                    {"scheme": s, "count": c, "percentage": (c / total_s) * 100.0}
                    for s, c in scheme_counter.most_common()
                ]
            ).set_index("scheme")
            save_table(sch_df, tables_dir, "url_scheme_frequencies.csv")
    else:
        print("[WARN] Column 'urls' not found; skipping domain/scheme analysis.")


def analyze_attachments(df: pd.DataFrame, tables_dir: Path, figs_dir: Path):
    """
    Analyze attachments:

      1) Email-level stats:
         - Histogram of attachments_count
         - Descriptive stats for attachments_count and attachments_total_size
         - attachment_statistics.csv

      2) Attachment-type frequencies:
         - attachment_type_frequencies.csv
         - attachment_types_bar.(png|eps)

      3) Attachment-unit size stats (USANDO attachments_sizes):
         - attachment_unit_size_statistics.csv
         - attachment_unit_size_histogram.(png|eps)
    """

    # ---------------------------------------------------------------------
    # 1) Email-level: histogram and stats for count / total_size
    # ---------------------------------------------------------------------
    has_count = "attachments_count" in df.columns
    has_total_size = "attachments_total_size" in df.columns

    # Histogram: number of attachments per email
    if has_count:
        count_series = _safe_numeric_series(df, "attachments_count")
        fig, ax = plt.subplots()
        if not count_series.dropna().empty:
            ax.hist(count_series.dropna(), bins=30, color=HIST_COLOR, edgecolor="black")
            _format_axes(
                ax,
                "Attachments per email",
                "Number of attachments",
                "Frequency",
            )
        else:
            ax.text(0.5, 0.5, "No valid attachments_count", ha="center", va="center")
            ax.set_axis_off()
        plt.tight_layout()
        save_figure(fig, figs_dir, "attachment_count_histogram")

    # Stats table for count and total_size at email level
    records = []

    if has_count:
        s = _safe_numeric_series(df, "attachments_count")
        if not s.dropna().empty:
            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            rec = {
                "metric": "attachments_count_all",
                "n": float(desc["count"]),
                "mean": float(desc["mean"]),
                "std": float(desc["std"]),
                "min": float(desc["min"]),
                "q25": float(desc["25%"]),
                "median": float(desc["50%"]),
                "q75": float(desc["75%"]),
                "max": float(desc["max"]),
            }
            # Emails with at least one attachment
            non_zero = s[s > 0]
            if not non_zero.empty:
                desc_nz = non_zero.describe(percentiles=[0.25, 0.5, 0.75])
                rec.update(
                    {
                        "n_with_attachments": float(desc_nz["count"]),
                        "mean_with_attachments": float(desc_nz["mean"]),
                        "std_with_attachments": float(desc_nz["std"]),
                        "min_with_attachments": float(desc_nz["min"]),
                        "q25_with_attachments": float(desc_nz["25%"]),
                        "median_with_attachments": float(desc_nz["50%"]),
                        "q75_with_attachments": float(desc_nz["75%"]),
                        "max_with_attachments": float(desc_nz["max"]),
                    }
                )
            else:
                rec["n_with_attachments"] = 0.0
            records.append(rec)

    if has_total_size:
        s = _safe_numeric_series(df, "attachments_total_size")
        if not s.dropna().empty:
            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            rec = {
                "metric": "attachments_total_size_all",
                "n": float(desc["count"]),
                "mean": float(desc["mean"]),
                "std": float(desc["std"]),
                "min": float(desc["min"]),
                "q25": float(desc["25%"]),
                "median": float(desc["50%"]),
                "q75": float(desc["75%"]),
                "max": float(desc["max"]),
            }
            # Emails with >=1 attachment (using attachments_count if available)
            if has_count:
                mask_has = _safe_numeric_series(df, "attachments_count").fillna(0) > 0
                s_nz = s[mask_has].dropna()
            else:
                s_nz = s[s > 0].dropna()

            if not s_nz.empty:
                desc_nz = s_nz.describe(percentiles=[0.25, 0.5, 0.75])
                rec.update(
                    {
                        "n_with_attachments": float(desc_nz["count"]),
                        "mean_with_attachments": float(desc_nz["mean"]),
                        "std_with_attachments": float(desc_nz["std"]),
                        "min_with_attachments": float(desc_nz["min"]),
                        "q25_with_attachments": float(desc_nz["25%"]),
                        "median_with_attachments": float(desc_nz["50%"]),
                        "q75_with_attachments": float(desc_nz["75%"]),
                        "max_with_attachments": float(desc_nz["max"]),
                    }
                )
            else:
                rec["n_with_attachments"] = 0.0
            records.append(rec)

    if records:
        stats_df = pd.DataFrame.from_records(records)
        save_table(stats_df, tables_dir, "attachment_statistics.csv")

    # ---------------------------------------------------------------------
    # 2) Attachment type/extension frequencies
    # ---------------------------------------------------------------------
    if "attachments_types" in df.columns:
        all_exts: list[str] = []

        for val in df["attachments_types"].dropna():
            if not isinstance(val, str):
                continue
            parsed = None
            # Try JSON first
            try:
                parsed = json.loads(val)
            except Exception:
                parsed = None

            if isinstance(parsed, list):
                for x in parsed:
                    t = str(x).strip().lower()
                    if not t:
                        continue
                    if t.startswith("."):
                        t = t[1:]
                    all_exts.append(t)
            else:
                # Fallback: split by comma / whitespace
                tokens = re.split(r"[,\s]+", val)
                for t in tokens:
                    t = t.strip().lower()
                    if not t:
                        continue
                    if t.startswith("."):
                        t = t[1:]
                    if 0 < len(t) <= 10:
                        all_exts.append(t)

        if all_exts:
            vc = pd.Series(all_exts).value_counts()
            total = vc.sum()
            ext_df = pd.DataFrame(
                {
                    "attachment_type": vc.index,
                    "count": vc.values,
                    "percentage": vc.values / total * 100.0,
                }
            ).set_index("attachment_type")
            save_table(ext_df, tables_dir, "attachment_type_frequencies.csv")

            # Top-20 types for bar chart
            top = vc.head(20)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(
                top.index,
                top.values,
                color=BAR_COLOR,
                edgecolor="black",
            )
            _format_axes(
                ax,
                "Attachment types (top 20)",
                "Type / extension",
                "Number of occurrences",
            )
            plt.xticks(rotation=60, ha="right")
            plt.tight_layout()
            save_figure(fig, figs_dir, "attachment_types_bar")

    # ---------------------------------------------------------------------
    # 3) Per-attachment unit size analysis (attachments_sizes)
    # ---------------------------------------------------------------------
    if "attachments_sizes" in df.columns:
        all_sizes: list[float] = []

        for val in df["attachments_sizes"].dropna():
            if isinstance(val, (int, float)):
                all_sizes.append(float(val))
                continue

            if not isinstance(val, str):
                continue

            parsed = None
            # Try JSON
            try:
                parsed = json.loads(val)
            except Exception:
                parsed = None

            if isinstance(parsed, list):
                for x in parsed:
                    try:
                        all_sizes.append(float(x))
                    except (TypeError, ValueError):
                        continue
            else:
                # Fallback: split by comma / whitespace
                tokens = re.split(r"[,\s]+", val)
                for t in tokens:
                    t = t.strip()
                    if not t:
                        continue
                    try:
                        all_sizes.append(float(t))
                    except (TypeError, ValueError):
                        continue

        if all_sizes:
            s_sizes = pd.Series(all_sizes, dtype="float64")
            desc = s_sizes.describe(percentiles=[0.25, 0.5, 0.75])

            rows = []
            for stat_name, value in desc.items():
                rows.append(
                    {
                        "stat": str(stat_name),
                        "attachment_unit_size": float(value),
                    }
                )

            unit_df = pd.DataFrame.from_records(rows)
            save_table(unit_df, tables_dir, "attachment_unit_size_statistics.csv")

            # Histogram of unit attachment sizes
            fig, ax = plt.subplots()
            ax.hist(s_sizes.dropna(), bins=40, color=HIST_COLOR, edgecolor="black")
            _format_axes(
                ax,
                "Attachment unit size distribution",
                "Attachment size (same units as dataset)",
                "Frequency",
            )
            plt.tight_layout()
            save_figure(fig, figs_dir, "attachment_unit_size_histogram")


def analyze_email_length(df: pd.DataFrame, tables_dir: Path, figs_dir: Path):
    """
    Compute character and word length statistics for subject and body.

    Outputs:
        - email_length_statistics.csv
        - subject_length_histogram.(png|eps)  (word count)
        - body_length_histogram.(png|eps)     (word count)
        - subject_char_length_histogram.(png|eps)
        - body_char_length_histogram.(png|eps)
    """
    df = _add_length_features(df)

    cols = [
        "subject_char_len",
        "body_char_len",
        "subject_word_len",
        "body_word_len",
    ]
    stats = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats.rename(
        columns={
            "count": "n",
            "mean": "mean",
            "std": "std",
            "min": "min",
            "25%": "q25",
            "50%": "median",
            "75%": "q75",
            "max": "max",
        },
        inplace=True,
    )

    save_table(stats, tables_dir, "email_length_statistics.csv")

    # Histograms for word length
    for col, title, fname in [
        ("subject_word_len", "Subject Length (Words)", "subject_length_histogram.png"),
        ("body_word_len", "Body Length (Words)", "body_length_histogram.png"),
    ]:
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=40, color=HIST_COLOR, edgecolor="black")
        _format_axes(ax, title, "Number of words", "Frequency")
        plt.tight_layout()
        save_figure(fig, figs_dir, fname.replace(".png", ""))

    # Histograms for character length
    for col, title, fname in [
        (
            "subject_char_len",
            "Subject Length (Characters)",
            "subject_char_length_histogram.png",
        ),
        ("body_char_len", "Body Length (Characters)", "body_char_length_histogram.png"),
    ]:
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=40, color=HIST_COLOR, edgecolor="black")
        _format_axes(ax, title, "Number of characters", "Frequency")
        plt.tight_layout()
        save_figure(fig, figs_dir, fname.replace(".png", ""))


def analyze_hops(df: pd.DataFrame, tables_dir: Path, figs_dir: Path):
    """
    Analyse the hops_count field (email routing complexity).

    Outputs:
        - hops_statistics.csv
        - hops_count_histogram.(png|eps)
    """
    if "hops_count" not in df.columns:
        print("[WARN] Column 'hops_count' not found; skipping hops analysis.")
        return

    hops_series = _safe_numeric_series(df, "hops_count")

    # -------------------------
    # Descriptive statistics table
    # -------------------------
    desc = hops_series.describe(percentiles=[0.25, 0.5, 0.75])

    rows = []
    for stat_name, value in desc.items():
        rows.append(
            {
                "stat": str(stat_name),
                "hops_count": float(value),
            }
        )

    hops_stats = pd.DataFrame.from_records(rows)
    save_table(hops_stats, tables_dir, "hops_statistics.csv")

    # -------------------------
    # Histogram
    # -------------------------
    fig, ax = plt.subplots()
    ax.hist(hops_series.dropna(), bins=30, color=HIST_COLOR, edgecolor="black")
    _format_axes(ax, "Number of Hops per Email", "Hops count", "Frequency")
    plt.tight_layout()
    save_figure(fig, figs_dir, "hops_count_histogram")
    print(f"[INFO] Saved figure: {figs_dir}")


def analyze_temporal_distribution(df: pd.DataFrame, tables_dir: Path, figs_dir: Path):
    """
    Temporal distribution of emails based on the `date` field.

    Outputs:
      - emails_over_time_year_month.csv   (year_month, email_count)
      - emails_over_time_year_month.(png|eps)   (time series of email volume)
    """
    df_parsed = _parse_dates(df)
    if "date_parsed" not in df_parsed.columns:
        print(
            "[WARN] No 'date_parsed' column after _parse_dates; skipping temporal analysis."
        )
        return

    date_col = df_parsed["date_parsed"]
    valid = date_col.dropna()

    if valid.empty:
        print("[WARN] No valid dates found; skipping temporal analysis.")
        return

    # Group by year-month period
    period = valid.dt.to_period("M")
    counts = period.value_counts().sort_index()

    temporal_df = counts.to_frame(name="email_count")
    temporal_df.index.name = "year_month"

    save_table(temporal_df, tables_dir, "emails_over_time_year_month.csv")

    # Figure: email volume over time (year-month)
    fig, ax = plt.subplots(figsize=(8, 4))
    x_vals = temporal_df.index.astype(str)
    ax.plot(
        x_vals,
        temporal_df["email_count"],
        marker="o",
        linestyle="-",
        color=LINE_COLOR,
    )
    _format_axes(
        ax,
        "Email Volume Over Time (Year-Month)",
        "Year-Month",
        "Number of emails",
    )

    # Reduce x-axis ticks when there are many periods
    if len(x_vals) > 12:
        step = max(1, int(np.ceil(len(x_vals) / 12)))
        tick_positions = np.arange(0, len(x_vals), step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [x_vals[i] for i in tick_positions],
            rotation=45,
            ha="right",
        )
    else:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    save_figure(fig, figs_dir, "emails_over_time_year_month")


def analyze_pop_descriptive_and_reliability(
    df: pd.DataFrame, tables_dir: Path, figs_dir: Path
):
    """
    Descriptive statistics and inter-rater reliability for persuasion principles.

    - If per-annotator columns (principle_A/B/C) exist, they are used to compute:
        * Descriptive statistics per rater
        * ICC(2,1) and Spearman correlations AB/AC/BC
    - If only the aggregated column exists, only global descriptives are reported
      and reliability metrics are set to NaN.
    - If no columns are found for a principle, it is skipped.

    Outputs:
        - pop_descriptive_statistics_by_rater.csv
        - pop_reliability_icc_and_correlations.csv
        - pop_mean_intensity_by_principle.(png|eps)
        - pop_icc_by_principle.(png|eps)
    """
    desc_records = []
    reliability_records = []

    for principle in PRINCIPLES:
        aggregated_exists = principle in df.columns
        cols_raters_all = [f"{principle}_A", f"{principle}_B", f"{principle}_C"]
        existing_raters = [c for c in cols_raters_all if c in df.columns]

        # If neither raters nor aggregated column exist, skip this principle
        if not existing_raters and not aggregated_exists:
            print(f"[WARN] No POP columns found for principle '{principle}'; skipping.")
            continue

        # ------------------------------------------------------
        # 1) Descriptive statistics per rater
        # ------------------------------------------------------
        for rater_col in existing_raters:
            s = pd.to_numeric(df[rater_col], errors="coerce")
            if s.notna().sum() == 0:
                continue

            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            desc_records.append(
                {
                    "principle": principle,
                    "rater": rater_col.split("_")[-1],
                    "n": float(desc["count"]),
                    "mean": float(desc["mean"]),
                    "std": float(desc["std"]),
                    "min": float(desc["min"]),
                    "q25": float(desc["25%"]),
                    "median": float(desc["50%"]),
                    "q75": float(desc["75%"]),
                    "max": float(desc["max"]),
                }
            )

        # ------------------------------------------------------
        # 2) Reliability (ICC and Spearman between raters)
        #    Only meaningful when >= 2 raters are available.
        # ------------------------------------------------------
        icc_value = np.nan
        r_AB = np.nan
        r_AC = np.nan
        r_BC = np.nan
        n_items = 0

        if len(existing_raters) >= 2:
            data_matrix = np.vstack(
                [
                    pd.to_numeric(df[c], errors="coerce").to_numpy()
                    for c in existing_raters
                ]
            ).T
            icc_value = compute_icc_2_1(data_matrix)
            n_items = int((~np.isnan(data_matrix).any(axis=1)).sum())

            def _safe_spearman(col1: str, col2: str) -> float:
                """Compute a Spearman correlation while handling invalid or degenerate inputs safely."""
                if col1 in df.columns and col2 in df.columns:
                    s1 = pd.to_numeric(df[col1], errors="coerce")
                    s2 = pd.to_numeric(df[col2], errors="coerce")
                    return float(s1.corr(s2, method="spearman"))
                return float("nan")

            # Compute all pairwise Spearman correlations; missing pairs return NaN
            r_AB = _safe_spearman(f"{principle}_A", f"{principle}_B")
            r_AC = _safe_spearman(f"{principle}_A", f"{principle}_C")
            r_BC = _safe_spearman(f"{principle}_B", f"{principle}_C")

        # If there is any data (raters or aggregated), record a reliability row
        if existing_raters or aggregated_exists:
            reliability_records.append(
                {
                    "principle": principle,
                    "n_items": n_items,
                    "icc_2_1": icc_value,
                    "spearman_AB": r_AB,
                    "spearman_AC": r_AC,
                    "spearman_BC": r_BC,
                }
            )

    # ----------------------------------------------------------
    # 3) Save tables
    # ----------------------------------------------------------
    if desc_records:
        desc_df = pd.DataFrame.from_records(desc_records)
        desc_df = desc_df.sort_values(by=["principle", "rater"])
        save_table(desc_df, tables_dir, "pop_descriptive_statistics_by_rater.csv")

    if reliability_records:
        rel_df = pd.DataFrame.from_records(reliability_records)
        rel_df = rel_df.sort_values(by="principle")
        save_table(rel_df, tables_dir, "pop_reliability_icc_and_correlations.csv")

    # ----------------------------------------------------------
    # 4) Bar chart: mean intensity per principle
    #    Uses aggregated column if available; otherwise averages raters.
    # ----------------------------------------------------------
    mean_by_principle = []
    for principle in PRINCIPLES:
        col_resolved = resolve_principle_column(df, principle)
        if col_resolved is not None:
            s = pd.to_numeric(df[col_resolved], errors="coerce")
        else:
            cols_raters = [f"{principle}_A", f"{principle}_B", f"{principle}_C"]
            existing = [c for c in cols_raters if c in df.columns]
            if not existing:
                continue
            s = pd.concat(
                [pd.to_numeric(df[c], errors="coerce") for c in existing], axis=1
            ).mean(axis=1)

        if s.notna().sum() == 0:
            continue

        mean_by_principle.append(
            {
                "principle": principle,
                "mean_intensity": float(s.mean()),
            }
        )

    if mean_by_principle:
        mean_df = pd.DataFrame(mean_by_principle).set_index("principle")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            mean_df.index,
            mean_df["mean_intensity"],
            color=BAR_COLOR,
            edgecolor="black",
        )
        _format_axes(
            ax, "Mean PoP Intensity by Principle", "Principle", "Mean intensity"
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        save_figure(fig, figs_dir, "pop_mean_intensity_by_principle")

    # ----------------------------------------------------------
    # 5) Bar chart: ICC(2,1) per principle
    # ----------------------------------------------------------
    if reliability_records:
        rel_plot_df = pd.DataFrame.from_records(reliability_records).set_index(
            "principle"
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            rel_plot_df.index,
            rel_plot_df["icc_2_1"],
            color=BAR_COLOR,
            edgecolor="black",
        )
        _format_axes(
            ax,
            "Inter-rater Reliability (ICC(2,1)) by Principle",
            "Principle",
            "ICC(2,1)",
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        save_figure(fig, figs_dir, "pop_icc_by_principle")


def analyze_pop_justifications(df: pd.DataFrame, tables_dir: Path):
    """
    Analyse the textual justification columns for persuasion principles (justif_*).

    For each justif_<principle>_<rater> column computes:
      - n            : total rows
      - n_non_empty  : rows with non-empty text
      - pct_non_empty
      - mean_chars / median_chars
      - mean_words / median_words

    Outputs:
      - pop_justifications_overview.csv
    """

    # Columns starting with "justif_"
    justif_cols = [c for c in df.columns if c.startswith("justif_")]
    if not justif_cols:
        print(
            "[INFO] No 'justif_*' columns found; skipping POP justifications analysis."
        )
        return

    records = []

    for col in justif_cols:
        # Expected column format: justif_<principle>_<rater>
        parts = col.split("_")
        if len(parts) < 3 or parts[0] != "justif":
            # Formato raro, igual lo registramos pero con principle/rater desconocidos
            principle_raw = "_".join(parts[1:-1]) if len(parts) > 1 else ""
            rater = parts[-1] if parts else ""
        else:
            rater = parts[-1]
            principle_raw = "_".join(parts[1:-1])

        # Resolve principle alias (e.g. commitement_... -> commitment_...)
        principle = PRINCIPLE_ALIASES.get(principle_raw, principle_raw)

        s = df[col].astype("string")

        n_total = int(len(s))
        if n_total == 0:
            continue

        non_empty_mask = s.notna() & (s.str.strip() != "")
        n_non_empty = int(non_empty_mask.sum())
        pct_non_empty = (n_non_empty / n_total) * 100.0 if n_total > 0 else 0.0

        if n_non_empty > 0:
            s_ne = s[non_empty_mask]
            char_len = s_ne.str.len()
            word_len = s_ne.str.split().str.len()

            mean_chars = float(char_len.mean())
            median_chars = float(char_len.median())
            mean_words = float(word_len.mean())
            median_words = float(word_len.median())
        else:
            mean_chars = np.nan
            median_chars = np.nan
            mean_words = np.nan
            median_words = np.nan

        records.append(
            {
                "column": col,
                "principle": principle,
                "principle_raw": principle_raw,
                "rater": rater,
                "n": n_total,
                "n_non_empty": n_non_empty,
                "pct_non_empty": pct_non_empty,
                "mean_chars": mean_chars,
                "median_chars": median_chars,
                "mean_words": mean_words,
                "median_words": median_words,
            }
        )

    if not records:
        print("[INFO] No non-empty POP justification entries found.")
        return

    justif_df = pd.DataFrame.from_records(records)

    # Sort by principle and rater for readability
    justif_df = justif_df.sort_values(by=["principle", "rater", "column"])

    save_table(justif_df, tables_dir, "pop_justifications_overview.csv")


def analyze_pop_vs_label(df: pd.DataFrame, tables_dir: Path, figs_dir: Path):
    """
    Analyse the association between persuasion intensity and the Label column
    using non-parametric tests.

    - Uses aggregated columns; if absent, averages annotator columns A/B/C.
    - Two labels: Mann-Whitney U + Cliff's delta.
    - Three or more labels: Kruskal-Wallis + eta².

    Outputs:
        - pop_label_association_tests.csv
        - pop_intensity_by_label_<principle>.(png|eps)
    """
    if "Label" not in df.columns:
        return

    label_series = df["Label"]
    test_records = []

    for principle in PRINCIPLES:
        # 1) Resolve intensity column for this principle
        col_resolved = resolve_principle_column(df, principle)
        if col_resolved is not None:
            intensity = pd.to_numeric(df[col_resolved], errors="coerce")
        else:
            cols_raters = [f"{principle}_A", f"{principle}_B", f"{principle}_C"]
            existing = [c for c in cols_raters if c in df.columns]
            if not existing:
                continue
            intensity = pd.concat(
                [pd.to_numeric(df[c], errors="coerce") for c in existing], axis=1
            ).mean(axis=1)

        tmp = pd.DataFrame({"Label": label_series, "intensity": intensity}).dropna()
        if tmp.shape[0] < 3:
            continue

        # 2) Group by raw label value
        labels_sorted = sorted(tmp["Label"].unique(), key=lambda x: str(x))
        data_per_label = [
            tmp[tmp["Label"] == lab]["intensity"] for lab in labels_sorted
        ]
        data_per_label = [s for s in data_per_label if not s.empty]

        if len(data_per_label) < 2:
            continue

        # 3) Statistical tests
        if len(data_per_label) == 2:
            group_a, group_b = data_per_label
            u_stat, p_val = mannwhitneyu(group_a, group_b, alternative="two-sided")
            n1, n2 = len(group_a), len(group_b)
            cliffs_delta = (2 * u_stat / (n1 * n2)) - 1 if n1 > 0 and n2 > 0 else np.nan
            test_records.append(
                {
                    "principle": principle,
                    "test": "mannwhitneyu",
                    "statistic": float(u_stat),
                    "p_value": float(p_val),
                    "effect_size": float(cliffs_delta),
                    "effect_size_name": "cliffs_delta",
                    "n_labels": 2,
                    "n_valid": int(len(group_a) + len(group_b)),
                }
            )
        else:
            h_stat, p_val = kruskal(*data_per_label)
            n_total = sum(len(g) for g in data_per_label)
            eta_sq = h_stat / (n_total - 1) if n_total > 1 else np.nan
            test_records.append(
                {
                    "principle": principle,
                    "test": "kruskal",
                    "statistic": float(h_stat),
                    "p_value": float(p_val),
                    "effect_size": float(eta_sq),
                    "effect_size_name": "eta_squared",
                    "n_labels": len(data_per_label),
                    "n_valid": int(n_total),
                }
            )

        # 4) Figure: boxplot per label using semantic label names
        mapping = _build_label_semantic_mapping(tmp["Label"])
        tick_labels = [mapping.get(lab, str(lab)) for lab in labels_sorted]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(
            [tmp[tmp["Label"] == lab]["intensity"] for lab in labels_sorted],
            tick_labels=tick_labels,
            patch_artist=True,
        )
        ax.set_title(f"PoP intensity by label - {principle}")
        ax.set_xlabel("Label")
        ax.set_ylabel("Intensity")

        # Consistent styling
        for spine in ["top", "right"]:
            if spine in ax.spines:
                ax.spines[spine].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", rotation=45)

        base_name = f"pop_intensity_by_label_{_sanitize_for_filename(principle)}"
        save_figure(fig, figs_dir, base_name)

    # 5) Save results table
    if test_records:
        tests_df = pd.DataFrame.from_records(test_records)
        save_table(tests_df, tables_dir, "pop_label_association_tests.csv")


def _infer_column_role(name: str, series: pd.Series) -> str:
    """
    Infer the high-level role of a column purely from data characteristics.

    Possible roles: 'numeric', 'categorical', 'text', 'list_like', 'unknown'.
    No semantic assumptions are made about the column name.
    """
    s = series.dropna()
    if s.empty:
        return "unknown"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Heuristic for list-like columns (urls, attachments_types, …)
    low_name = name.lower()
    if any(
        key in low_name
        for key in ["urls", "url_list", "attachments_types", "types", "list", "hops"]
    ):
        return "list_like"

    # Convert to string for basic inspection
    s_str = s.astype(str)
    avg_len = s_str.str.len().mean()
    n_unique = s_str.nunique()

    # Categorical: few unique values and short strings
    if n_unique <= 50 and avg_len <= 50:
        return "categorical"

    # Texto libre
    return "text"


def generate_data_dictionary(df: pd.DataFrame, tables_dir: Path):
    """
    Generate an automatic data dictionary without adding manual descriptions.

    Outputs:
        - data_dictionary_auto.csv  (one row per column with type, role, and stats)
    """
    records = []
    for col in df.columns:
        series = df[col]
        role = _infer_column_role(col, series)
        n_total = len(series)
        n_non_null = series.notna().sum()
        n_null = n_total - n_non_null
        pct_null = (n_null / n_total) * 100.0 if n_total > 0 else np.nan
        n_unique = series.nunique(dropna=True)

        min_val = np.nan
        max_val = np.nan
        if pd.api.types.is_numeric_dtype(series):
            s_num = pd.to_numeric(series, errors="coerce")
            if s_num.notna().any():
                min_val = float(s_num.min())
                max_val = float(s_num.max())

        # First non-null value, truncated to 120 characters
        example_value = ""
        first_non_null = series.dropna().iloc[0] if series.dropna().shape[0] > 0 else ""
        if isinstance(first_non_null, (list, dict)):
            example_value = str(first_non_null)
        else:
            example_value = str(first_non_null)
        if len(example_value) > 120:
            example_value = example_value[:117] + "..."

        records.append(
            {
                "column_name": col,
                "pandas_dtype": str(series.dtype),
                "inferred_role": role,
                "n_non_null": int(n_non_null),
                "n_null": int(n_null),
                "pct_null": pct_null,
                "n_unique": int(n_unique),
                "min": min_val,
                "max": max_val,
                "example_value": example_value,
                # Fields to be filled in manually:
                "description": "",
                "units": "",
                "allowed_values": "",
            }
        )

    dict_df = pd.DataFrame.from_records(records)
    dict_df = dict_df.sort_values(by="column_name")
    save_table(dict_df, tables_dir, "data_dictionary_auto.csv")


def analyze_univariate_columns(df: pd.DataFrame, tables_dir: Path):
    """
    Generate univariate summaries for all columns.

    Outputs:
      - univariate_numeric_summary.csv    (numeric columns)
      - univariate_categorical_summary.csv  (text / categorical / list-like columns)
    """
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        role = _infer_column_role(col, df[col])
        if role == "numeric":
            numeric_cols.append(col)
        elif role in ["categorical", "text", "list_like"]:
            categorical_cols.append(col)

    # ---- NUMERIC ----
    num_records = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        n = s.notna().sum()
        if n == 0:
            continue

        desc = s.describe(percentiles=[0.25, 0.5, 0.75])
        mean = float(desc["mean"])
        std = float(desc["std"]) if not math.isnan(desc["std"]) else np.nan
        var = float(std**2) if std == std else np.nan  # NaN safe

        skew = float(s.skew()) if n > 2 else np.nan
        kurt = float(s.kurtosis()) if n > 3 else np.nan

        n_zero = int((s == 0).sum())
        n_total = len(s)
        n_missing = int(s.isna().sum())
        pct_missing = (n_missing / n_total) * 100.0 if n_total > 0 else np.nan

        # Outliers via IQR fence
        q1 = float(desc["25%"])
        q3 = float(desc["75%"])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((s < lower) | (s > upper)).sum())
        pct_outliers = (n_outliers / n) * 100.0 if n > 0 else np.nan

        num_records.append(
            {
                "column_name": col,
                "n": int(n),
                "mean": mean,
                "std": std,
                "var": var,
                "min": float(desc["min"]),
                "q25": float(desc["25%"]),
                "median": float(desc["50%"]),
                "q75": float(desc["75%"]),
                "max": float(desc["max"]),
                "skewness": skew,
                "kurtosis": kurt,
                "n_zero": n_zero,
                "n_missing": n_missing,
                "pct_missing": pct_missing,
                "n_outliers_iqr": n_outliers,
                "pct_outliers_iqr": pct_outliers,
            }
        )

    if num_records:
        num_df = pd.DataFrame.from_records(num_records)
        num_df = num_df.sort_values(by="column_name")
        save_table(num_df, tables_dir, "univariate_numeric_summary.csv")

    # ---- CATEGORICAL / TEXT ----
    cat_records = []
    for col in categorical_cols:
        s = df[col].astype(str)
        n_total = len(s)
        n_non_null = (df[col].notna()).sum()
        n_null = n_total - n_non_null
        pct_null = (n_null / n_total) * 100.0 if n_total > 0 else np.nan

        value_counts = s.value_counts(dropna=False)
        n_unique = int(value_counts.shape[0])

        top_value = str(value_counts.index[0]) if n_unique > 0 else ""
        top_count = int(value_counts.iloc[0]) if n_unique > 0 else 0
        top_pct = (top_count / n_total) * 100.0 if n_total > 0 else np.nan

        # Shannon entropy (base 2)
        freqs = value_counts.to_numpy(dtype=float)
        freqs = freqs / freqs.sum() if freqs.sum() > 0 else freqs
        entropy = (
            float(-(freqs * np.log2(freqs + 1e-12)).sum()) if n_unique > 0 else np.nan
        )

        cat_records.append(
            {
                "column_name": col,
                "n_total": int(n_total),
                "n_non_null": int(n_non_null),
                "n_null": int(n_null),
                "pct_null": pct_null,
                "n_unique": n_unique,
                "top_value": top_value[:100],
                "top_count": top_count,
                "top_pct": top_pct,
                "entropy_bits": entropy,
            }
        )

    if cat_records:
        cat_df = pd.DataFrame.from_records(cat_records)
        cat_df = cat_df.sort_values(by="column_name")
        save_table(cat_df, tables_dir, "univariate_categorical_summary.csv")


# =============================================================================
# DATA QUALITY ASSESSMENT — PART 1: Missingness Analysis
# =============================================================================


def analyze_missingness(df: pd.DataFrame, tables_dir: Path, figs_dir: Path):
    """
    Analyse data quality in terms of missing values.

    Outputs:
        - missingness_summary.csv
        - missingness_heatmap.(png|eps)
        - missingness_barplot.(png|eps)
    """
    # -----------------------------
    # 1) Percentage of NA per column
    # -----------------------------
    n_rows = len(df)
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / n_rows) * 100

    summary = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_pct": missing_pct,
        }
    ).sort_values(by="missing_pct", ascending=False)

    # Save as a flat table with explicit column name
    summary_out = summary.reset_index().rename(columns={"index": "column"})
    summary_out.to_csv(tables_dir / "missingness_summary.csv", index=False)

    # -----------------------------
    # 2) NA heatmap (columns with at least one missing value)
    # -----------------------------
    # Only include columns that actually have missing values
    cols_with_na = [c for c in df.columns if df[c].isna().any()]

    if len(cols_with_na) > 0:
        subset = df[cols_with_na]
        if subset.shape[0] > 2000:
            subset = subset.sample(n=2000, random_state=42)

        fig, ax = plt.subplots(figsize=(10, max(4, len(cols_with_na) * 0.25)))
        sns.heatmap(subset.isna(), cmap=["#ffffff", "#08306B"], cbar=False, ax=ax)
        ax.set_title("Missingness Matrix (NA Heatmap) - Columns with Missing Values")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        plt.tight_layout()
        save_figure(fig, figs_dir, "missingness_heatmap")
    # -----------------------------
    # 3) Bar plot: percentage of NA per column
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=summary.index,
        y=summary["missing_pct"],
        color=BAR_COLOR,
        edgecolor="black",
        ax=ax,
    )
    ax.set_title("Percentage of Missing Values per Column")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Missing (%)")
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    save_figure(fig, figs_dir, "missingness_barplot")


# =============================================================================
# BASELINE TEXT CLASSIFICATION (TF-IDF + Logistic Regression / Random Forest)
# =============================================================================


def run_baseline_text_models(df: pd.DataFrame, tables_dir: Path, figs_dir: Path):
    """
    Train baseline text classifiers on the Label column (subject + body as features).

    Models:
      - Logistic Regression (TF-IDF, unigrams + bigrams)
      - Random Forest (TF-IDF, unigrams + bigrams)

    Outputs:
      - tables/baseline_metrics.csv
      - figures/baseline_confusion_logreg.(png|eps)
      - figures/baseline_confusion_rf.(png|eps)
      - figures/baseline_roc_logreg.(png|eps)   (binary labels only)
      - figures/baseline_roc_rf.(png|eps)       (binary labels only)
    """
    if "Label" not in df.columns:
        print("[WARN] Column 'Label' not found; skipping baseline models.")
        return

    # ---------- 1) Prepare text features ----------
    df_work = df.copy()

    if "subject" in df_work.columns:
        df_work["subject_str"] = df_work["subject"].fillna("").astype(str)
    else:
        df_work["subject_str"] = ""

    if "body" in df_work.columns:
        df_work["body_str"] = df_work["body"].fillna("").astype(str)
    else:
        df_work["body_str"] = ""

    df_work["text_concat"] = (
        df_work["subject_str"] + " " + df_work["body_str"]
    ).str.strip()

    # Drop rows with no label or empty text
    mask_valid = df_work["Label"].notna() & df_work["text_concat"].str.len().gt(0)
    df_valid = df_work.loc[mask_valid].copy()

    if df_valid.shape[0] < 50:
        print(
            f"[WARN] Not enough valid samples for baseline (n={df_valid.shape[0]}). Skipping."
        )
        return

    texts = df_valid["text_concat"].tolist()
    labels_raw = df_valid["Label"]

    # ---------- 2) Encode label ----------
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)

    n_classes = len(le.classes_)
    if n_classes < 2:
        print(
            "[WARN] Only one class present in Label after filtering; skipping baseline."
        )
        return

    # ---------- 3) Train/test split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ---------- 4) TF-IDF vectorisation ----------
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        lowercase=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ---------- 5) Define models ----------
    models = {
        "logreg": LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }

    metrics_records = []

    for name, clf in models.items():
        print(f"[INFO] Training baseline model: {name}")
        clf.fit(X_train_vec, y_train)

        y_pred = clf.predict(X_test_vec)

        # Basic metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        # Probabilities for ROC-AUC (if supported by the model)
        roc_auc = np.nan
        y_proba = None
        has_proba = hasattr(clf, "predict_proba")
        if has_proba:
            y_proba = clf.predict_proba(X_test_vec)
            try:
                if n_classes == 2:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(
                        y_test, y_proba, multi_class="ovr", average="macro"
                    )
            except Exception as e:
                print(f"[WARN] Could not compute ROC-AUC for {name}: {e}")
                roc_auc = np.nan

        metrics_records.append(
            {
                "model": name,
                "n_test": int(len(y_test)),
                "n_classes": int(n_classes),
                "accuracy": acc,
                "precision_macro": prec_macro,
                "recall_macro": rec_macro,
                "f1_macro": f1_macro,
                "precision_weighted": prec_weight,
                "recall_weighted": rec_weight,
                "f1_weighted": f1_weight,
                "roc_auc": roc_auc,
            }
        )

        # ---------- Confusion matrix ----------

        cm = confusion_matrix(y_test, y_pred, labels=range(n_classes))
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        im = ax_cm.imshow(cm, interpolation="nearest", aspect="auto")
        ax_cm.set_title(f"Confusion matrix ({name})")
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")
        tick_labels = [str(c) for c in le.classes_]
        ax_cm.set_xticks(range(n_classes))
        ax_cm.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax_cm.set_yticks(range(n_classes))
        ax_cm.set_yticklabels(tick_labels, fontsize=8)

        # Annotate cells with counts
        for i in range(n_classes):
            for j in range(n_classes):
                ax_cm.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                )

        fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
        plt.tight_layout()
        save_figure(fig_cm, figs_dir, f"baseline_confusion_{name}")

        # ---------- ROC curve (binary labels only, requires predict_proba) ----------
        if has_proba and n_classes == 2:
            try:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                ax_roc.plot(
                    fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=1.5
                )
                ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
                ax_roc.set_xlabel("False positive rate")
                ax_roc.set_ylabel("True positive rate")
                ax_roc.set_title(f"ROC curve ({name})")
                ax_roc.legend(loc="lower right", fontsize=8)
                ax_roc.grid(True, linestyle="--", alpha=0.4)
                plt.tight_layout()
                save_figure(fig_roc, figs_dir, f"baseline_roc_{name}")
            except Exception as e:
                print(f"[WARN] Could not plot ROC curve for {name}: {e}")

    # ---------- Save metrics table ----------
    if metrics_records:
        metrics_df = pd.DataFrame.from_records(metrics_records)
        metrics_df = metrics_df.set_index("model")
        save_table(metrics_df, tables_dir, "baseline_metrics.csv")
    else:
        print("[WARN] No baseline metrics computed.")


# =============================================================================
# LABEL SANITY CHECK
# =============================================================================


def analyze_label_sanity(df: pd.DataFrame, tables_dir: Path, figs_dir: Path):
    """
    Perform a structural sanity check on the Label column.

    Outputs:
      - label_sanity_check.csv         : comparative structural statistics by label
      - label_sanity_anomalies.csv     : rows with suspicious label assignments
      - label_sanity_structural_boxplots.(png|eps)
      - label_sanity_anomaly_scatter.(png|eps)
    """

    if "Label" not in df.columns:
        print("[WARN] Column 'Label' not found; skipping label sanity check.")
        return

    # Work on a copy
    df_work = df.copy()

    # Normalise label to coherent semantic values (phishing / legitimate / other)
    df_work["_Label_norm"] = _normalize_label_semantic_series(df_work["Label"])

    # Ensure length features are computed
    df_work = _add_length_features(df_work)

    # --------- 1) Comparative structural statistics by label ---------
    structural_cols = [
        "url_count",
        "attachments_count",
        "attachments_total_size",
        "hops_count",
        "subject_word_len",
        "body_word_len",
        "subject_char_len",
        "body_char_len",
        # Aggregated PoP columns if present; otherwise silently ignored below
        "authority",
        "social_proof",
        "liking_similarity_deception",
        "commitement_integrity_reciprocation",
        "distraction",
    ]

    existing_cols = [c for c in structural_cols if c in df_work.columns]

    # Convert all structural columns to numeric to avoid object-dtype issues
    for col in existing_cols:
        df_work[col] = pd.to_numeric(df_work[col], errors="coerce")

    if existing_cols:
        summary = df_work.groupby("_Label_norm")[existing_cols].agg(
            ["mean", "median", "std", "min", "max"]
        )
        summary.to_csv(tables_dir / "label_sanity_check.csv")
        print("[INFO] Saved: label_sanity_check.csv")
    else:
        print("[WARN] No structural numeric columns found for label sanity summary.")
        summary = None

    # --------- 2) Anomaly detection ---------
    anomalies = []

    # Determine the PoP intensity source per principle:
    # - Use aggregated column (e.g. "authority") if available.
    # - Otherwise fall back to individual rater columns (*_A/_B/_C).
    pop_sources = []
    for principle in PRINCIPLES:
        agg_col = resolve_principle_column(df_work, principle)
        rater_cols = [
            c
            for c in [
                f"{principle}_A",
                f"{principle}_B",
                f"{principle}_C",
            ]
            if c in df_work.columns
        ]
        if agg_col is not None or rater_cols:
            pop_sources.append((principle, agg_col, rater_cols))

    for idx, row in df_work.iterrows():
        label = row["_Label_norm"]

        # Structural values are already numeric (float/NaN)
        urlc = row.get("url_count", np.nan)
        attc = row.get("attachments_count", np.nan)

        # PoP values from aggregated or rater columns

        pop_vals = []
        for principle, agg_col, rater_cols in pop_sources:
            # Prioridad: columna agregada si existe; si no, raters
            if agg_col is not None:
                cols_to_use = [agg_col]
            else:
                cols_to_use = rater_cols

            for col_name in cols_to_use:
                v = row.get(col_name, np.nan)
                if pd.isna(v):
                    pop_vals.append(np.nan)
                else:
                    try:
                        pop_vals.append(float(v))
                    except (TypeError, ValueError):
                        pop_vals.append(np.nan)

        if not pop_vals or all(np.isnan(pop_vals)):
            mean_pop = np.nan
        else:
            mean_pop = float(np.nanmean(pop_vals))

        suspicious = False
        reasons = []

        # Case 1: phishing with 0 URLs and 0 attachments
        if label == "phishing" and (
            (pd.isna(urlc) or urlc == 0) and (pd.isna(attc) or attc == 0)
        ):
            suspicious = True
            reasons.append("Phishing label but no URLs and no attachments")

        # Case 2: legitimate email with unusually many URLs
        if label in ("legit", "benign", "ham", "legitimate"):
            if not pd.isna(urlc) and urlc >= 5:
                suspicious = True
                reasons.append(f"Legit label but unusually high URL count: {urlc}")

        # Case 3: legitimate email with extremely high PoP score
        if (
            label in ("legit", "ham", "legitimate")
            and not np.isnan(mean_pop)
            and mean_pop > 0.8
        ):
            suspicious = True
            reasons.append(f"Legit label but mean POP too high: {mean_pop:.2f}")

        if suspicious:
            anomalies.append(
                {
                    "index": idx,
                    "label": label,
                    "url_count": urlc,
                    "attachments_count": attc,
                    "mean_pop": mean_pop,
                    "reason": "; ".join(reasons),
                }
            )

    anomalies_df = pd.DataFrame(anomalies)
    anomalies_df.to_csv(tables_dir / "label_sanity_anomalies.csv", index=False)
    print("[INFO] Saved: label_sanity_anomalies.csv")

    # --------- 3) Structural boxplots by label ---------
    unique_labels = sorted(df_work["_Label_norm"].dropna().unique(), key=str)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    selected = ["url_count", "attachments_count", "subject_word_len", "body_word_len"]
    selected = [c for c in selected if c in df_work.columns]

    for ax, col in zip(axes.flatten(), selected):
        ax.set_title(f"{col} by label")

        data = [
            df_work.loc[df_work["_Label_norm"] == lbl, col].dropna()
            for lbl in unique_labels
        ]

        if all(len(d) == 0 for d in data):
            ax.text(0.5, 0.5, f"No data for {col}", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.boxplot(
                data,
                patch_artist=True,
                tick_labels=unique_labels,  # requiere matplotlib >= 3.9
            )
            ax.tick_params(axis="x", rotation=45)
            for spine_name in ["top", "right"]:
                spine_obj = ax.spines.get(spine_name)
                if spine_obj is not None:
                    spine_obj.set_visible(False)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Turn off unused subplots
    if len(selected) < len(axes.flatten()):
        for ax in axes.flatten()[len(selected) :]:
            ax.axis("off")

    plt.tight_layout()
    save_figure(fig, figs_dir, "label_sanity_structural_boxplots")
    plt.close(fig)

    # --------- 4) Anomaly scatter plot ---------
    if (
        not anomalies_df.empty
        and "url_count" in df_work.columns
        and "attachments_count" in df_work.columns
    ):
        fig, ax = plt.subplots(figsize=(6, 5))

        # Normal points
        ax.scatter(
            df_work["url_count"],
            df_work["attachments_count"],
            alpha=0.3,
            color="#4393c3",
            label="Normal",
        )

        # Anomalous points
        ax.scatter(
            anomalies_df["url_count"],
            anomalies_df["attachments_count"],
            color="#d73027",
            label="Anomalies",
            s=60,
        )

        ax.set_xlabel("url_count")
        ax.set_ylabel("attachments_count")
        ax.set_title("Label Anomaly Scatter")
        ax.legend()

        save_figure(fig, figs_dir, "label_sanity_anomaly_scatter")
        plt.close(fig)
    else:
        print("[INFO] No anomalies or missing structural columns for anomaly scatter.")


# =============================================================================
# TEXTUAL EXAMPLES (DISPLAY DE EJEMPLOS)
# =============================================================================


def _make_body_excerpt(text: str, max_len: int = 400) -> str:
    """Return an excerpt of the body field, truncated to max_len characters."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def generate_textual_examples(
    df: pd.DataFrame, tables_dir: Path, max_per_label: int = 8, max_per_pop: int = 8
):
    """
    Generate tables with textual examples from the dataset for qualitative inspection.

    Outputs:
        - examples_by_label.csv   (representative emails per label)
        - examples_high_pop.csv   (emails with the highest PoP intensity per principle)
    """

    if "Label" not in df.columns:
        print("[WARN] Column 'Label' not found; skipping textual examples.")
        return

    has_hash = "hash" in df.columns
    has_subject = "subject" in df.columns
    has_body = "body" in df.columns
    has_date = "date" in df.columns

    # -------------------------------------------------------------------------
    # 1) Examples per label
    # -------------------------------------------------------------------------
    examples_rows = []

    labels_unique = df["Label"].dropna().astype(str).str.strip().unique()

    for lab in sorted(labels_unique, key=lambda x: str(x)):
        subset = df[df["Label"].astype(str).str.strip() == lab].copy()
        if subset.empty:
            continue

        # Deterministic order: sort by hash if available, otherwise by index
        if has_hash:
            subset = subset.sort_values(by="hash")
        else:
            subset = subset.sort_index()

        subset = subset.head(max_per_label)

        for idx, row in subset.iterrows():
            record = {
                "label": str(lab),
                "index": idx,
            }
            if has_hash:
                record["hash"] = row["hash"]
            if has_date:
                record["date"] = row["date"]
            if has_subject:
                record["subject"] = (
                    row["subject"]
                    if isinstance(row["subject"], str)
                    else str(row["subject"])
                )
            if has_body:
                record["body_excerpt"] = _make_body_excerpt(row["body"], max_len=400)

            # Useful structural fields if present
            for col in [
                "url_count",
                "attachments_count",
                "attachments_total_size",
                "hops_count",
            ]:
                if col in df.columns:
                    record[col] = row.get(col, None)

            # Intensidades agregadas si existen
            for col in [
                "authority",
                "social_proof",
                "liking_similarity_deception",
                "commitement_integrity_reciprocation",
                "distraction",
            ]:
                if col in df.columns:
                    record[col] = row.get(col, None)

            examples_rows.append(record)

    if examples_rows:
        examples_df = pd.DataFrame(examples_rows)
        examples_df.to_csv(tables_dir / "examples_by_label.csv", index=False)
        print("[INFO] Saved: examples_by_label.csv")
    else:
        print("[WARN] No textual examples by label produced (no labels?).")

    # -------------------------------------------------------------------------
    # 2) High-intensity PoP examples
    # -------------------------------------------------------------------------
    pop_principles = [
        "authority",
        "social_proof",
        "liking_similarity_deception",
        "commitement_integrity_reciprocation",
        "distraction",
    ]

    high_pop_rows = []

    for principle in pop_principles:
        if principle not in df.columns:
            continue

        s = pd.to_numeric(df[principle], errors="coerce")
        valid_idx = s.dropna().index
        if len(valid_idx) == 0:
            continue

        # Sort by intensity descending
        top_idx = s.loc[valid_idx].sort_values(ascending=False).head(max_per_pop).index

        for idx in top_idx:
            row = df.loc[idx]
            record = {
                "principle": principle,
                "intensity": float(s.loc[idx]),
                "index": idx,
                "label": row.get("Label", None),
            }
            if has_hash:
                record["hash"] = row.get("hash", None)
            if has_date:
                record["date"] = row.get("date", None)
            if has_subject:
                record["subject"] = (
                    row["subject"]
                    if isinstance(row["subject"], str)
                    else str(row["subject"])
                )
            if has_body:
                record["body_excerpt"] = _make_body_excerpt(row["body"], max_len=400)

            # Include structural features when present
            for col in [
                "url_count",
                "attachments_count",
                "attachments_total_size",
                "hops_count",
            ]:
                if col in df.columns:
                    record[col] = row.get(col, None)

            high_pop_rows.append(record)

    if high_pop_rows:
        high_pop_df = pd.DataFrame(high_pop_rows)
        high_pop_df.to_csv(tables_dir / "examples_high_pop.csv", index=False)
        print("[INFO] Saved: examples_high_pop.csv")
    else:
        print("[WARN] No high-intensity PoP examples produced (no PoP columns?).")


# =============================================================================
# EXPORTABLE DATASET SCHEMA
# =============================================================================


def generate_dataset_schema(
    df: pd.DataFrame, tables_dir: Path, dataset_name: str = "SpaPhish"
):
    """
    Generate an exportable dataset schema derived purely from the data.

    Outputs:
      - dataset_schema_tabular.csv   (one row per column with type metadata)
      - dataset_schema_json.json     (JSON schema-like description)
    """

    records = []
    for col in df.columns:
        series = df[col]
        n_total = len(series)
        n_non_null = int(series.notna().sum())
        n_null = n_total - n_non_null
        pct_null = (n_null / n_total * 100.0) if n_total > 0 else np.nan
        n_unique = int(series.nunique(dropna=True))

        pandas_dtype = str(series.dtype)
        role = _infer_column_role(col, series)

        is_nullable = n_null > 0

        min_val = None
        max_val = None
        allowed_values = None

        # Numeric: record min and max
        if pd.api.types.is_numeric_dtype(series):
            s_num = pd.to_numeric(series, errors="coerce")
            if s_num.notna().any():
                min_val = float(s_num.min())
                max_val = float(s_num.max())

        # Categorical: list allowed values if cardinality is manageable
        if role == "categorical":
            vc = series.dropna().astype(str).value_counts()
            if vc.shape[0] <= 50:
                allowed_values = sorted(vc.index.tolist())

        # Ejemplo
        example_value = ""
        non_null = series.dropna()
        if not non_null.empty:
            example_value = str(non_null.iloc[0])
            if len(example_value) > 120:
                example_value = example_value[:117] + "..."

        rec = {
            "name": col,
            "pandas_dtype": pandas_dtype,
            "inferred_role": role,
            "is_nullable": bool(is_nullable),
            "n_total": int(n_total),
            "n_non_null": n_non_null,
            "n_null": n_null,
            "pct_null": pct_null,
            "n_unique": n_unique,
            "min": min_val,
            "max": max_val,
            "example_value": example_value,
            "allowed_values": allowed_values,
        }
        records.append(rec)

    # --- Tabular CSV ---
    schema_df = pd.DataFrame.from_records(records)
    schema_df = schema_df.sort_values(by="name")
    schema_csv_path = tables_dir / "dataset_schema_tabular.csv"
    schema_df.to_csv(schema_csv_path, index=False)
    print(f"[INFO] Saved: {schema_csv_path}")

    # --- JSON schema ---
    schema_json = {
        "dataset_name": dataset_name,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "fields": records,
    }
    schema_json_path = tables_dir / "dataset_schema_json.json"
    schema_json_path.write_text(
        json.dumps(schema_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[INFO] Saved: {schema_json_path}")


# =============================================================================
# POTENTIAL BIAS ANALYSIS (LABEL vs STRUCTURAL/TEMPORAL FEATURES)
# =============================================================================


def analyze_potential_bias(df: pd.DataFrame, tables_dir: Path, figs_dir: Path):
    """
    Analyse potential dataset biases with respect to the Label column.

      - Association between Label and numeric structural / PoP features.
      - Temporal drift of label distribution.
      - Standardised mean difference (SMD) of features for a primary label vs rest.

    Outputs:
      - bias_feature_label_association.csv
      - bias_temporal_label_distribution.csv
      - bias_bucket_chi2.csv
      - bias_effect_sizes_label_vs_rest.csv
      - bias_effect_sizes_<label>_vs_rest.(png|eps)
    """

    if "Label" not in df.columns:
        print("[WARN] Column 'Label' not found; skipping potential bias analysis.")
        return

    # Normalise label to string
    label_raw = df["Label"]
    label_norm = label_raw.astype(str).str.strip()
    df = df.copy()
    df["_Label_norm"] = label_norm

    # Ensure length features are computed
    df = _add_length_features(df)

    # -----------------------------------------
    # 1) Association between Label and numeric features
    # -----------------------------------------
    candidate_numeric = [
        "url_count",
        "attachments_count",
        "attachments_total_size",
        "hops_count",
        "subject_word_len",
        "body_word_len",
        "subject_char_len",
        "body_char_len",
        "authority",
        "social_proof",
        "liking_similarity_deception",
        "commitement_integrity_reciprocation",
        "distraction",
    ]
    numeric_features = [c for c in candidate_numeric if c in df.columns]

    assoc_records = []
    unique_labels = sorted(label_norm.dropna().unique(), key=lambda x: str(x))

    for feat in numeric_features:
        s = pd.to_numeric(df[feat], errors="coerce")
        # grupos por label
        groups = []
        group_labels = []
        for lab in unique_labels:
            g = s[df["_Label_norm"] == lab].dropna()
            if len(g) > 0:
                groups.append(g)
                group_labels.append(lab)

        # Kruskal–Wallis global
        if len(groups) >= 2:
            try:
                stat_kw, p_kw = kruskal(*groups)
            except Exception:
                stat_kw, p_kw = (np.nan, np.nan)
        else:
            stat_kw, p_kw = (np.nan, np.nan)

        for lab in unique_labels:
            g = s[df["_Label_norm"] == lab].dropna()
            if g.empty:
                continue
            rest = s[df["_Label_norm"] != lab].dropna()

            n_lab = int(len(g))
            n_rest = int(len(rest))

            mean_lab = float(g.mean())
            std_lab = float(g.std(ddof=1)) if n_lab > 1 else np.nan

            if n_rest > 1:
                mean_rest = float(rest.mean())
                var_lab = float(g.var(ddof=1)) if n_lab > 1 else np.nan
                var_rest = float(rest.var(ddof=1))
                denom = n_lab + n_rest - 2
                if denom > 0 and not math.isnan(var_lab):
                    pooled_sd = math.sqrt(
                        ((n_lab - 1) * var_lab + (n_rest - 1) * var_rest) / denom
                    )
                else:
                    pooled_sd = np.nan
                if pooled_sd and pooled_sd == pooled_sd and pooled_sd != 0.0:
                    smd = (mean_lab - mean_rest) / pooled_sd
                else:
                    smd = np.nan
            else:
                smd = np.nan

            assoc_records.append(
                {
                    "feature": feat,
                    "label": lab,
                    "n_label": n_lab,
                    "n_rest": n_rest,
                    "mean_label": mean_lab,
                    "std_label": std_lab,
                    "smd_label_vs_rest": smd,
                    "kruskal_p_global": p_kw,
                }
            )

    if assoc_records:
        assoc_df = pd.DataFrame.from_records(assoc_records)
        assoc_df = assoc_df.sort_values(by=["feature", "label"])
        out_path = tables_dir / "bias_feature_label_association.csv"
        assoc_df.to_csv(out_path, index=False)
        print(f"[INFO] Saved: {out_path}")
    else:
        print("[WARN] No numeric features available for bias analysis.")

    # -----------------------------------------
    # 2) Drift temporal de Label
    # -----------------------------------------
    df = _parse_dates(df)
    if "date_parsed" in df.columns and df["date_parsed"].notna().any():
        tmp = df[["date_parsed", "_Label_norm"]].dropna()
        tmp["year_month"] = tmp["date_parsed"].dt.to_period("M").astype(str)

        crosstab = tmp.pivot_table(
            index="year_month",
            columns="_Label_norm",
            values="date_parsed",
            aggfunc="count",
            fill_value=0,
        )

        # proporciones por mes
        prop = crosstab.div(crosstab.sum(axis=1), axis=0)

        # guardamos ambas
        dist_path = tables_dir / "bias_temporal_label_distribution.csv"
        crosstab.to_csv(dist_path)
        print(f"[INFO] Saved: {dist_path}")

        prop_path = tables_dir / "bias_temporal_label_proportions.csv"
        prop.to_csv(prop_path)
        print(f"[INFO] Saved: {prop_path}")
    else:
        print("[WARN] No valid dates for temporal bias analysis.")

    # -----------------------------------------
    # 3) Buckets simples + Chi² con Label
    # -----------------------------------------
    bucket_records = []

    # Buckets de url_count
    if "url_count" in df.columns:
        s = pd.to_numeric(df["url_count"], errors="coerce")
        url_bucket = pd.cut(
            s, bins=[-0.5, 0.5, 1.5, 3.5, np.inf], labels=["0", "1", "2-3", "4+"]
        )
        df["_url_bucket"] = url_bucket.astype(str)

        ct = pd.crosstab(df["_Label_norm"], df["_url_bucket"], dropna=False)
        if ct.size > 0:
            try:
                chi2, p, dof, _ = chi2_contingency(ct)
            except Exception:
                chi2, p, dof = (np.nan, np.nan, np.nan)
            bucket_records.append(
                {
                    "variable": "url_bucket",
                    "chi2": chi2,
                    "p_value": p,
                    "dof": dof,
                }
            )

    # Buckets de attachments_count
    if "attachments_count" in df.columns:
        s = pd.to_numeric(df["attachments_count"], errors="coerce")
        att_bucket = pd.cut(s, bins=[-0.5, 0.5, 1.5, np.inf], labels=["0", "1", "2+"])
        df["_att_bucket"] = att_bucket.astype(str)

        ct = pd.crosstab(df["_Label_norm"], df["_att_bucket"], dropna=False)
        if ct.size > 0:
            try:
                chi2, p, dof, _ = chi2_contingency(ct)
            except Exception:
                chi2, p, dof = (np.nan, np.nan, np.nan)
            bucket_records.append(
                {
                    "variable": "attachments_bucket",
                    "chi2": chi2,
                    "p_value": p,
                    "dof": dof,
                }
            )

    # Buckets de longitud (body_word_len)
    if "body_word_len" in df.columns:
        s = pd.to_numeric(df["body_word_len"], errors="coerce")
        quantiles = (
            s.quantile([0.33, 0.66]).values.tolist() if s.notna().any() else None
        )
        if quantiles and quantiles[0] < quantiles[1]:
            bins = [-0.5, quantiles[0], quantiles[1], np.inf]
            labels = ["short", "medium", "long"]
            len_bucket = pd.cut(s, bins=bins, labels=labels)
            df["_body_len_bucket"] = len_bucket.astype(str)

            ct = pd.crosstab(df["_Label_norm"], df["_body_len_bucket"], dropna=False)
            if ct.size > 0:
                try:
                    chi2, p, dof, _ = chi2_contingency(ct)
                except Exception:
                    chi2, p, dof = (np.nan, np.nan, np.nan)
                bucket_records.append(
                    {
                        "variable": "body_length_bucket",
                        "chi2": chi2,
                        "p_value": p,
                        "dof": dof,
                    }
                )

    if bucket_records:
        bucket_df = pd.DataFrame.from_records(bucket_records)
        out_path = tables_dir / "bias_bucket_chi2.csv"
        bucket_df.to_csv(out_path, index=False)
        print(f"[INFO] Saved: {out_path}")
    else:
        print("[WARN] No bucket-based chi² analyses produced.")

    # -----------------------------------------
    # 4) Figura de efecto estandarizado (un label vs resto)
    # -----------------------------------------
    if assoc_records:
        assoc_df = pd.DataFrame.from_records(assoc_records)
        labels_available = sorted(assoc_df["label"].unique(), key=lambda x: str(x))

        # Elegimos un label "principal": phishing si existe, si no el primero
        target_label = None
        for lab in labels_available:
            if "phish" in str(lab).lower():
                target_label = lab
                break
        if target_label is None and labels_available:
            target_label = labels_available[0]

        if target_label is not None:
            eff_df = assoc_df[assoc_df["label"] == target_label].copy()
            eff_df["abs_smd"] = eff_df["smd_label_vs_rest"].abs()

            eff_df = eff_df.sort_values(by="abs_smd", ascending=False)

            # Guardamos tabla
            eff_path = tables_dir / "bias_effect_sizes_label_vs_rest.csv"
            eff_df.to_csv(eff_path, index=False)
            print(f"[INFO] Saved: {eff_path}")

            # Figura
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(
                eff_df["feature"], eff_df["abs_smd"], color=BAR_COLOR, edgecolor="black"
            )
            ax.set_ylabel("Absolute SMD (label vs rest)")
            ax.set_xlabel("Feature")
            ax.set_title(f"Effect sizes for label '{target_label}' vs rest")
            ax.tick_params(axis="x", rotation=45)
            for spine in ["top", "right"]:
                if spine in ax.spines:
                    ax.spines[spine].set_visible(False)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

            base_name = (
                f"bias_effect_sizes_{_sanitize_for_filename(target_label)}_vs_rest"
            )
            save_figure(fig, figs_dir, base_name)
            plt.close(fig)
        else:
            print("[WARN] Could not determine a target label for effect-size figure.")


# =============================================================================
# Main
# =============================================================================


def main():
    # =========================================================================
    # 0. BASE PATHS & DATASET
    # =========================================================================
    # This script lives in analysis/, so the project root is one level up.
    """Run the SpaPhish analysis pipeline and return an exit code."""
    project_root = Path(__file__).resolve().parent.parent

    # Accept CSV (primary download format) or Excel
    data_path_csv = project_root / "data" / "Spaphish dataset - DiB.csv"
    data_path_xlsx = project_root / "data" / "Spaphish dataset - DiB.xlsx"
    if data_path_csv.exists():
        data_path = data_path_csv
    elif data_path_xlsx.exists():
        data_path = data_path_xlsx
    else:
        raise FileNotFoundError(
            f"Dataset not found. Expected one of:\n"
            f"  {data_path_csv}\n"
            f"  {data_path_xlsx}\n"
            "Download it from Mendeley Data and place it in the data/ folder."
        )

    output_base = project_root / "output"

    print(f"[INFO] Loading dataset from: {data_path}")
    df = load_dataset(data_path)
    print(f"[INFO] Dataset loaded with shape: {df.shape}")

    dirs = ensure_output_dirs(output_base)
    tables_dir = dirs["tables"]
    figs_dir = dirs["figures"]

    print(f"[INFO] Outputs will be stored under: {output_base}")

    # =========================================================================
    # 1. METADATA & BASIC SCHEMA
    # =========================================================================
    print("[INFO] Generating data dictionary...")
    generate_data_dictionary(df, tables_dir)

    print("[INFO] Analyzing univariate columns...")
    analyze_univariate_columns(df, tables_dir)

    print("[INFO] Generating dataset schema...")
    generate_dataset_schema(df, tables_dir, dataset_name="SpaPhish")

    # =========================================================================
    # 2. MISSINGNESS
    # =========================================================================
    print("[INFO] Analyzing missingness...")
    analyze_missingness(df, tables_dir, figs_dir)

    # =========================================================================
    # 3. STRUCTURAL EDA (LABEL + EMAIL STRUCTURE)
    # =========================================================================
    print("[INFO] Analyzing label distribution...")
    analyze_label_distribution(df, tables_dir, figs_dir)

    print("[INFO] Analyzing URLs...")
    analyze_urls(df, tables_dir, figs_dir)

    print("[INFO] Analyzing attachments...")
    analyze_attachments(df, tables_dir, figs_dir)

    print("[INFO] Analyzing email length...")
    analyze_email_length(df, tables_dir, figs_dir)

    print("[INFO] Analyzing hops count...")
    analyze_hops(df, tables_dir, figs_dir)

    print("[INFO] Analyzing temporal distribution...")
    analyze_temporal_distribution(df, tables_dir, figs_dir)

    # =========================================================================
    # 4. PERSUASION PRINCIPLES (PoP): DESCRIPTIVES, RELIABILITY, LABEL, JUSTIFICATIONS
    # =========================================================================
    print("[INFO] Analyzing PoP descriptive statistics and reliability...")
    analyze_pop_descriptive_and_reliability(df, tables_dir, figs_dir)

    print("[INFO] Analyzing PoP vs Label...")
    analyze_pop_vs_label(df, tables_dir, figs_dir)

    print("[INFO] Analyzing PoP textual justifications (justif_*)...")
    analyze_pop_justifications(df, tables_dir)

    # =========================================================================
    # 5. BASELINES, SANITY CHECK, BIAS & TEXT EXAMPLES
    # =========================================================================
    print("[INFO] Running baseline text classification models...")
    run_baseline_text_models(df, tables_dir, figs_dir)

    print("[INFO] Running label sanity check...")
    analyze_label_sanity(df, tables_dir, figs_dir)

    print("[INFO] Running potential bias analysis...")
    analyze_potential_bias(df, tables_dir, figs_dir)

    print("[INFO] Generating textual examples...")
    generate_textual_examples(df, tables_dir)

    # =========================================================================
    # 6. GLOBAL OVERVIEW, WORDCLOUDS & OUTPUT MANIFEST
    # =========================================================================
    print("[INFO] Generating data summary figure (overview)...")
    generate_data_summary_figure(df, figs_dir)

    print("[INFO] Generating wordclouds...")
    generate_wordclouds(df, figs_dir)

    print("[INFO] Generating outputs manifest...")
    generate_outputs_manifest(tables_dir, figs_dir)

    print(f"[INFO] All outputs stored under: {output_base}")


if __name__ == "__main__":
    main()
