#!/usr/bin/env python
"""SpaPhish pipeline entry point.

This is the only public command-line entry point for the repository.
It exposes four descriptive subcommands:

* process - convert raw `.eml` messages into a structured CSV dataset
* analyze - generate tables and figures from an existing SpaPhish CSV
* report  - build HTML and PDF reports from precomputed analysis outputs
* all     - run the full pipeline end to end

Each stage can be executed independently, or chained through `all`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_STRUCTURED_CSV = DEFAULT_OUTPUT_DIR / "processing" / "spaphish_eml_dataset.csv"
DEFAULT_CSV = PROJECT_ROOT / "data" / "Spaphish dataset - DiB.csv"
DEFAULT_XLSX = PROJECT_ROOT / "data" / "Spaphish dataset - DiB.xlsx"


def resolve_dataset_csv(csv_path: Optional[Path]) -> Path:
    """Resolve the SpaPhish CSV input, falling back to the published Excel file."""

    candidates: list[Path] = []
    if csv_path is not None:
        candidates.append(csv_path)
    else:
        candidates.extend([DEFAULT_STRUCTURED_CSV, DEFAULT_CSV, DEFAULT_XLSX])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    expected = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "No SpaPhish dataset file was found. Expected one of:\n"
        f"{expected}\n"
        "Download the dataset from Mendeley Data and place it under data/."
    )


def build_html_report(output_dir: Path) -> tuple[Path, Path]:
    """Build the HTML report and its companion PDF from precomputed outputs."""

    from reporting import report_html

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    if not tables_dir.exists() or not figures_dir.exists():
        raise FileNotFoundError(
            "The analysis outputs are missing. Run the analyze stage before building reports."
        )

    try:
        from weasyprint import HTML
    except ImportError as exc:
        raise ImportError(
            "WeasyPrint is required to build the HTML report. Install the project dependencies first."
        ) from exc

    report = report_html.Report(
        title="SpaPhish Dataset - Descriptive Report",
        subtitle="Static HTML/PDF report generated from pre-computed tables and figures under output/.",
    )

    report_html.add_section_eda_overview(report, tables_dir, figures_dir)
    report_html.add_section_data_quality(report, tables_dir, figures_dir)
    report_html.add_section_label_distribution(report, tables_dir, figures_dir)
    report_html.add_section_urls(report, tables_dir, figures_dir)
    report_html.add_section_attachments(report, tables_dir, figures_dir)
    report_html.add_section_length(report, tables_dir, figures_dir)
    report_html.add_section_hops(report, tables_dir, figures_dir)
    report_html.add_section_temporal(report, tables_dir, figures_dir)
    report_html.add_section_pop_reliability(report, tables_dir, figures_dir)
    report_html.add_section_pop_vs_label(report, tables_dir, figures_dir)
    report_html.add_section_baseline(report, tables_dir, figures_dir)
    report_html.add_section_label_sanity(report, tables_dir, figures_dir)
    report_html.add_section_wordclouds(report, figures_dir)
    report_html.add_section_textual_examples(report, tables_dir)
    report_html.add_section_bias_analysis(report, tables_dir, figures_dir)
    report_html.add_section_value_range(report, tables_dir)
    report_html.add_section_outputs_manifest(report, tables_dir)
    report_html.add_section_dataset_schema(report, tables_dir)

    html_path = output_dir / "index_updated.html"
    pdf_path = output_dir / "SpaPhish_Report_UPDATED.pdf"

    html_path.write_text(report.render_html(), encoding="utf-8")
    print(f"[OK] HTML written: {html_path}")

    HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    print(f"[OK] PDF written:  {pdf_path}")
    return html_path, pdf_path


def build_standalone_pdf(output_dir: Path, pdf_path: Optional[Path] = None) -> Path:
    """Build the standalone ReportLab PDF report from precomputed outputs."""

    from reporting import report_pdf

    pdf_path = pdf_path or (output_dir / "SpaPhish_Report.pdf")
    report_pdf.build_pdf(output_dir=output_dir, pdf_path=pdf_path)
    return pdf_path


def run_process(args: argparse.Namespace) -> int:
    """Process raw `.eml` files and generate the structured SpaPhish CSV."""

    from processing.spaphish_processing import process_dataset, resolve_paths

    raw_dir = args.raw_dir or DEFAULT_RAW_DIR
    processed_dir = args.processed_dir or DEFAULT_PROCESSED_DIR
    processing_dir = args.output_dir / "processing"
    paths = resolve_paths(
        root=PROJECT_ROOT,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        output_dir=processing_dir,
    )

    print(f"[INFO] Raw email folder: {raw_dir}")
    print(f"[INFO] Processing output: {processing_dir}")
    try:
        process_dataset(
            paths=paths,
            enable_ocr=not args.no_ocr,
            ocr_lang=args.ocr_lang,
            max_images_per_email=args.max_images_per_email,
            max_ocr_text_chars=args.max_ocr_text_chars,
            allow_remote_images=args.allow_remote_images,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


def run_analyze(args: argparse.Namespace) -> int:
    """Generate tables and figures from an existing SpaPhish CSV."""

    try:
        from analysis.analyze_dataset import run_analysis

        data_path = resolve_dataset_csv(args.csv)
        print(f"[INFO] Analysis input: {data_path}")
        run_analysis(data_path=data_path, output_base=args.output_dir, dataset_name=args.dataset_name)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


def run_report(args: argparse.Namespace) -> int:
    """Generate report outputs from the analysis artifacts."""

    try:
        if args.report_format in {"html", "both"}:
            build_html_report(args.output_dir)
        if args.report_format in {"pdf", "both"}:
            build_standalone_pdf(args.output_dir)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


def run_all(args: argparse.Namespace) -> int:
    """Run processing, analysis, and reporting in sequence."""

    if args.csv is not None:
        try:
            from analysis.analyze_dataset import run_analysis

            data_path = resolve_dataset_csv(args.csv)
            print(f"[INFO] Skipping raw-email processing and starting from CSV: {data_path}")
            run_analysis(data_path=data_path, output_base=args.output_dir, dataset_name=args.dataset_name)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            return 1
    else:
        from analysis.analyze_dataset import run_analysis
        from processing.spaphish_processing import process_dataset, resolve_paths

        raw_dir = args.raw_dir or DEFAULT_RAW_DIR
        processed_dir = args.processed_dir or DEFAULT_PROCESSED_DIR
        processing_dir = args.output_dir / "processing"
        paths = resolve_paths(
            root=PROJECT_ROOT,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            output_dir=processing_dir,
        )

        print(f"[INFO] Raw email folder: {raw_dir}")
        print(f"[INFO] Processing output: {processing_dir}")
        try:
            process_dataset(
                paths=paths,
                enable_ocr=not args.no_ocr,
                ocr_lang=args.ocr_lang,
                max_images_per_email=args.max_images_per_email,
                max_ocr_text_chars=args.max_ocr_text_chars,
                allow_remote_images=args.allow_remote_images,
            )
            run_analysis(
                data_path=paths.dataset_csv,
                output_base=args.output_dir,
                dataset_name=args.dataset_name,
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            return 1

    try:
        if args.report_format in {"html", "both"}:
            build_html_report(args.output_dir)
        if args.report_format in {"pdf", "both"}:
            build_standalone_pdf(args.output_dir)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the SpaPhish command-line interface."""

    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "SpaPhish pipeline controller. Use the subcommands to process raw emails, analyze the CSV dataset, "
            "or generate report artifacts step by step."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py process --raw-dir data/raw\n"
            "  python main.py analyze --csv \"data/Spaphish dataset - DiB.csv\"\n"
            "  python main.py report --report-format both\n"
            "  python main.py all --raw-dir data/raw --report-format both"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser(
        "process",
        help="Convert raw `.eml` files into a structured SpaPhish CSV dataset.",
        description=(
            "Process a folder of raw `.eml` messages, deduplicate them by SHA-256, extract features, "
            "and write the structured CSV plus processing logs."
        ),
    )
    process_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Folder containing the raw `.eml` messages to process (default: data/raw).",
    )
    process_parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Folder where deduplicated `.eml` files will be stored (default: data/processed).",
    )
    process_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root folder for generated artifacts (default: output).",
    )
    process_parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR even when pytesseract and Tesseract are available.",
    )
    process_parser.add_argument(
        "--ocr-lang",
        default="eng+spa",
        help="Tesseract language code used for OCR, for example `eng+spa`.",
    )
    process_parser.add_argument(
        "--max-images-per-email",
        type=int,
        default=20,
        help="Maximum number of inline or embedded images to OCR per email.",
    )
    process_parser.add_argument(
        "--max-ocr-text-chars",
        type=int,
        default=20000,
        help="Maximum number of OCR text characters to keep per email.",
    )
    process_parser.add_argument(
        "--allow-remote-images",
        action="store_true",
        help="Allow remote image URLs inside HTML emails while extracting inline images.",
    )
    process_parser.set_defaults(func=run_process)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Generate descriptive tables and figures from the structured CSV produced by process.",
        description=(
            "Load the structured SpaPhish CSV produced by the processing stage, compute the descriptive analysis, "
            "and write the tables and figures to output/. You can also point to any previously generated structured CSV."
        ),
    )
    analyze_parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "Structured SpaPhish CSV file to analyze. If omitted, the command looks for the CSV generated by "
            "`process` and then falls back to a structured file under data/."
        ),
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root folder where analysis tables and figures will be written (default: output).",
    )
    analyze_parser.add_argument(
        "--dataset-name",
        default="SpaPhish",
        help="Human-readable dataset name used in exported schema tables.",
    )
    analyze_parser.set_defaults(func=run_analyze)

    report_parser = subparsers.add_parser(
        "report",
        help="Generate report artifacts from the tables and figures already in output/.",
        description=(
            "Build the publication-style report from precomputed analysis artifacts. "
            "You can generate the HTML report, the standalone PDF report, or both."
        ),
    )
    report_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root folder containing output/tables and output/figures (default: output).",
    )
    report_parser.add_argument(
        "--report-format",
        choices=("html", "pdf", "both"),
        default="both",
        help=(
            "Choose which report artifacts to build: html for the HTML report and companion PDF, "
            "pdf for the standalone PDF report, or both for all report outputs."
        ),
    )
    report_parser.set_defaults(func=run_report)

    all_parser = subparsers.add_parser(
        "all",
        help="Run processing, analysis, and reporting in one pass.",
        description=(
            "Execute the full SpaPhish pipeline end to end. Provide raw `.eml` files to process them first, "
            "or provide an existing CSV to start at the analysis stage."
        ),
    )
    source_group = all_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Folder containing raw `.eml` messages to process before analysis.",
    )
    source_group.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Existing SpaPhish CSV to start from the analysis stage.",
    )
    all_parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Folder where deduplicated `.eml` files will be stored when processing raw email.",
    )
    all_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root folder for all generated artifacts (default: output).",
    )
    all_parser.add_argument(
        "--dataset-name",
        default="SpaPhish",
        help="Human-readable dataset name used in exported schema tables.",
    )
    all_parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR even when pytesseract and Tesseract are available.",
    )
    all_parser.add_argument(
        "--ocr-lang",
        default="eng+spa",
        help="Tesseract language code used for OCR, for example `eng+spa`.",
    )
    all_parser.add_argument(
        "--max-images-per-email",
        type=int,
        default=20,
        help="Maximum number of inline or embedded images to OCR per email.",
    )
    all_parser.add_argument(
        "--max-ocr-text-chars",
        type=int,
        default=20000,
        help="Maximum number of OCR text characters to keep per email.",
    )
    all_parser.add_argument(
        "--allow-remote-images",
        action="store_true",
        help="Allow remote image URLs inside HTML emails while extracting inline images.",
    )
    all_parser.add_argument(
        "--report-format",
        choices=("html", "pdf", "both"),
        default="both",
        help=(
            "Choose which report artifacts to build after analysis: html for the HTML report and companion PDF, "
            "pdf for the standalone PDF report, or both for all report outputs."
        ),
    )
    all_parser.set_defaults(func=run_all)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Parse the command line and execute the selected pipeline stage."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
