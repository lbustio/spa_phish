# SpaPhish: Spanish Phishing Email Dataset Companion

This repository is the code companion for the **SpaPhish** dataset, a Spanish-language phishing email corpus published in *Data in Brief* (Elsevier) and distributed through Mendeley Data.

SpaPhish is a curated collection of **1,395 anonymized emails** spanning **2014 to 2025**. It includes:

* **731 phishing emails**
* **664 legitimate emails**
* raw subject and body text
* technical metadata such as URLs, routing depth, and attachment statistics
* persuasion annotations based on the **Principles of Persuasion** framework
* human-written justifications for the persuasion labels

The dataset was assembled from personal and institutional inboxes, then anonymized with manual redaction and controlled substitution so that the content remains readable while reducing re-identification risk.

The main value of SpaPhish is that it combines:

* native Spanish-language phishing data,
* a clean phishing-versus-legitimate distinction,
* technical email structure features,
* and psychological annotations for explainable analysis.

This repository provides the reproducible code needed to process the raw email files, analyze the CSV dataset, and generate publication-ready reports.

## About SpaPhish

SpaPhish was designed for research on phishing detection, social engineering, multilingual NLP, and explainable email analysis. It supports:

* phishing detection experiments in Spanish,
* hybrid text-plus-metadata modeling,
* analysis of attachment and URL patterns,
* longitudinal studies of phishing tactics,
* and annotation reliability studies for persuasion labels.

Each email record can contain:

* `subject` and `body`
* technical variables such as `url_count`, `urls`, `hops_count`, `attachments_count`, `attachments_types`, `attachments_total_size`, and `attachments_sizes`
* persuasion labels for:
  * Authority
  * Social Proof
  * Liking/Similarity/Deception
  * Commitment/Integrity/Reciprocation
  * Distraction
* brief justification text for the annotation decisions

## Repository Layout

```text
.
./analysis/
  analyze_dataset.py      descriptive analysis, tables, figures, quality checks
./docs/
  tareas.txt              project notes and historical task list
./processing/
  __init__.py             processing stage package marker
  spaphish_processing.py  raw email processing pipeline
./scripts/
  analyze_dataset.py      CLI entry point for analysis
  report.py               compatibility launcher for the HTML report
  report_html.py          CLI entry point for the HTML report
  report_pdf.py           CLI entry point for the PDF report
  spaphish.py             CLI entry point for raw email processing
  verify_project.py       repository validation and smoke-test utility
./reporting/
  report_html.py          HTML report generator
  report_pdf.py           PDF report generator
./notebooks/
  spaphish.ipynb          notebook launcher for the raw email pipeline
./tools/
  _build_notebook.py      notebook construction helper / legacy utility
./data/                   local dataset files downloaded from Mendeley Data
./output/                 generated tables, figures, HTML, and PDF reports
```

## Data Source

The dataset is available from Mendeley Data:

* Reserved DOI: `10.17632/hz2d6gz7pc.4`
* DOI link: https://doi.org/10.17632/hz2d6gz7pc.4
* Mendeley Data landing page: https://data.mendeley.com/datasets/hz2d6gz7pc/4

This repository does not version-control the dataset itself. After downloading it, place the CSV in `data/Spaphish dataset - DiB.csv`.

## Data Setup

The CSV is the primary file used by the analysis and reporting stages. An Excel version (`data/Spaphish dataset - DiB.xlsx`) is also accepted by the analysis script.

If you want to run the raw email processing notebook or script, place the source `.eml` files under `data/raw/`.

## Dependencies

Install the Python packages listed in `requirements.txt`.

If you prefer Conda, create the project environment with:

```bash
conda env create -f environment.yml
conda activate spaphish
```

Additional non-Python tools may be needed for the notebook:

* `tesseract-ocr` for OCR through `pytesseract`
* `Ghostscript` for embedding EPS figures in the PDF report
* `WeasyPrint` for turning the HTML report into PDF

## Full Pipeline

Run the steps below from the project root.

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the dataset in `data/`

Confirm that one of these files exists:

```text
data/Spaphish dataset - DiB.csv
data/Spaphish dataset - DiB.xlsx
```

### 4. Run the analysis stage

```bash
python scripts/analyze_dataset.py
```

This generates the tables and figures under `output/tables/` and `output/figures/`.

### 5. Build the HTML report

```bash
python scripts/report_html.py
```

This produces `output/index_updated.html` and the report assets referenced by the HTML page.

### 6. Build the PDF report

```bash
python scripts/report_pdf.py
```

This produces the PDF version of the report in `output/`.

### 7. Run the raw email processing stage

Run either `python scripts/spaphish.py` or open `notebooks/spaphish.ipynb`. Both call the same processing module under `processing/`.

The processing stage will:

* hash raw `.eml` files with SHA-256,
* deduplicate messages,
* extract structural and textual features,
* optionally run OCR on embedded images when `pytesseract` and Tesseract are available,
* export a cleaned dataset for downstream analysis.

## Validation

Use the validation script to confirm that the repository is structurally sound and that the main entry points can still run:

```bash
python scripts/verify_project.py
```

For the most useful result, run it from the `spaphish` Conda environment.

Add `--full` to run the heavier end-to-end checks. In full mode, the script will try to:

* run the analysis pipeline,
* build the HTML report,
* build the PDF report,
* and run the raw email processor if `data/raw/` contains `.eml` files.

The validation script is meant to catch broken imports, syntax issues, missing files, and execution regressions early.

## Dataset Fields

The CSV is organized around three layers:

1. Raw message content:
   * subject
   * body text
   * anonymized content
2. Technical indicators:
   * URL counts and URL lists
   * routing depth from `Received` headers
   * attachment counts, file types, and sizes
3. Psychological annotations:
   * binary persuasion labels for the five SpaPhish principles
   * annotator justifications

These fields make the dataset useful for both standard supervised learning and explainable research.

## Recommended Citation

If you use the dataset or this companion code in academic work, cite the SpaPhish Data in Brief article and reference the Mendeley Data record identified by the reserved DOI above.

## Output Structure

The pipeline writes its results to `output/`:

```text
output/
├── processing/                 # raw-email processing outputs
│   ├── spaphish_eml_dataset.csv
│   ├── duplicates.txt
│   ├── hashes.txt
│   ├── processing_errors.log
│   ├── ocr_errors.log
│   └── processing_summary.json
├── figures/                    # PNG and EPS figures
├── tables/                     # CSV summary tables and schema files
├── index_updated.html          # static HTML report
├── SpaPhish_Report.pdf         # PDF report
└── SpaPhish_Report_UPDATED.pdf # PDF produced by the HTML workflow
```

Key generated tables include:

* `outputs_manifest.csv`
* `dataset_schema_tabular.csv`
* `dataset_schema_json.json`
* `data_dictionary_auto.csv`
* `missingness_summary.csv`
* `label_distribution.csv`
* `baseline_metrics.csv`

Key generated figures include:

* `data_summary_overview.png`
* `label_distribution_bar.png`
* `missingness_heatmap.png`
* `emails_over_time_year_month.png`
* `baseline_roc_logreg.png`

## Notes for Mendeley Data

This repository is intended to be linked from the SpaPhish Mendeley Data entry as the companion code. The expected workflow is:

1. Download the dataset from Mendeley Data.
2. Place the CSV in `data/`.
3. Run the analysis stage.
4. Generate the report outputs.

Because the dataset is excluded from version control, users can clone the repository without receiving the data files, then add them locally before running the pipeline.
