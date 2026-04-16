"""Validate the SpaPhish repository structure and main execution paths."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CheckResult:
    """Result of a validation step."""

    status: str
    name: str
    details: str


def project_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[1]


def pass_result(name: str, details: str) -> CheckResult:
    """Create a passing validation result."""

    return CheckResult("PASS", name, details)


def warn_result(name: str, details: str) -> CheckResult:
    """Create a warning validation result."""

    return CheckResult("WARN", name, details)


def fail_result(name: str, details: str) -> CheckResult:
    """Create a failing validation result."""

    return CheckResult("FAIL", name, details)


def short_text(text: str, limit: int = 900) -> str:
    """Return a trimmed version of text for concise terminal output."""

    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def is_dependency_error_message(text: str) -> bool:
    """Return True when text looks like a missing-runtime-dependency error."""

    lower = text.lower()
    return (
        "modulenotfounderror" in lower
        or "importerror" in lower
        or "not installed" in lower
        or "install:" in lower
    )


def run_subprocess(command: Sequence[str], cwd: Path, timeout: float) -> CheckResult:
    """Run a subprocess and convert the outcome into a validation result."""

    label = " ".join(command[:3]) if len(command) >= 3 else " ".join(command)
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return fail_result(label, f"Command timed out after {timeout:.0f} seconds.")
    except OSError as exc:
        return fail_result(label, f"Could not start command: {exc}")

    if completed.returncode != 0:
        stdout = short_text(completed.stdout or "<empty>")
        stderr = short_text(completed.stderr or "<empty>")
        details = f"Exit code {completed.returncode}.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        if is_dependency_error_message(stdout) or is_dependency_error_message(stderr):
            return warn_result(label, details)
        return fail_result(label, details)

    stdout = short_text(completed.stdout or "<empty>")
    return pass_result(label, f"Exit code 0.\nSTDOUT:\n{stdout}")


def iter_python_files(root: Path) -> list[Path]:
    """Return the Python source files that should be syntax-checked."""

    candidates: list[Path] = []
    for relative in ("analysis", "processing", "reporting", "scripts", "tools"):
        base = root / relative
        if base.exists():
            candidates.extend(sorted(base.rglob("*.py")))
    return [path for path in candidates if "__pycache__" not in path.parts]


def check_project_layout(root: Path) -> list[CheckResult]:
    """Verify that the expected folders and compatibility files are present."""

    results: list[CheckResult] = []
    required_dirs = [
        "analysis",
        "data",
        "docs",
        "notebooks",
        "output",
        "processing",
        "reporting",
        "scripts",
        "tools",
    ]
    missing_dirs = [name for name in required_dirs if not (root / name).exists()]
    if missing_dirs:
        results.append(fail_result("project layout", f"Missing directories: {', '.join(missing_dirs)}"))
    else:
        results.append(pass_result("project layout", "All expected top-level directories are present."))

    required_files = [
        "analysis/__init__.py",
        "processing/__init__.py",
        "reporting/__init__.py",
        "README.md",
        "requirements.txt",
        "environment.yml",
        "data/README.md",
        "data/Spaphish dataset - DiB.csv",
    ]
    missing_files = [name for name in required_files if not (root / name).exists()]
    if missing_files:
        results.append(fail_result("required files", f"Missing files: {', '.join(missing_files)}"))
    else:
        results.append(pass_result("required files", "All required files are present."))

    legacy_root_files = [
        "analyze_dataset.py",
        "report.py",
        "report_html.py",
        "report_pdf.py",
        "spaphish.py",
        "spaphish.ipynb",
    ]
    still_present = [name for name in legacy_root_files if (root / name).exists()]
    if still_present:
        results.append(
            fail_result(
                "legacy root files",
                f"Unexpected legacy files still in the repository root: {', '.join(still_present)}",
            )
        )
    else:
        results.append(pass_result("legacy root files", "No legacy duplicates remain in the repository root."))

    return results


def check_python_syntax(root: Path) -> list[CheckResult]:
    """Parse all project Python files to confirm the code is syntactically valid."""

    results: list[CheckResult] = []
    files = iter_python_files(root)
    failures: list[str] = []
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(root)}: {exc.msg} at line {exc.lineno}")
    if failures:
        results.append(fail_result("python syntax", "\n".join(failures)))
    else:
        results.append(pass_result("python syntax", f"Parsed {len(files)} Python files successfully."))
    return results


def check_imports(root: Path) -> list[CheckResult]:
    """Import the canonical modules and CLI launchers."""

    results: list[CheckResult] = []
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    modules = [
        "analysis.analyze_dataset",
        "processing.spaphish_processing",
        "reporting.report_html",
        "reporting.report_pdf",
        "scripts.analyze_dataset",
        "scripts.report",
        "scripts.report_html",
        "scripts.report_pdf",
        "scripts.spaphish",
        "scripts.verify_project",
    ]

    failures: list[str] = []
    warnings: list[str] = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:  # pragma: no cover - surfaced in validation output
            warnings.append(f"{module_name}: {exc.__class__.__name__}: {exc}")
        except Exception as exc:  # pragma: no cover - surfaced in validation output
            failures.append(f"{module_name}: {exc.__class__.__name__}: {exc}")
    if failures:
        results.append(fail_result("imports", "\n".join(failures)))
    elif warnings:
        results.append(
            warn_result(
                "imports",
                "Some modules could not be imported because runtime dependencies are missing.\n"
                + "\n".join(warnings),
            )
        )
    else:
        results.append(pass_result("imports", f"Imported {len(modules)} modules successfully."))
    return results


def check_documentation(root: Path) -> list[CheckResult]:
    """Check that the dataset documentation points to the reserved DOI."""

    results: list[CheckResult] = []
    readme = (root / "README.md").read_text(encoding="utf-8")
    data_readme = (root / "data" / "README.md").read_text(encoding="utf-8")
    environment = (root / "environment.yml").read_text(encoding="utf-8")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    notebook = json.loads((root / "notebooks" / "spaphish.ipynb").read_text(encoding="utf-8"))

    required_readme_tokens = [
        "10.17632/hz2d6gz7pc.4",
        "https://data.mendeley.com/datasets/hz2d6gz7pc/4",
        "conda activate spaphish",
        "scripts/verify_project.py",
    ]
    missing_readme = [token for token in required_readme_tokens if token not in readme]
    if missing_readme:
        results.append(
            fail_result(
                "README",
                f"Missing expected README tokens: {', '.join(missing_readme)}",
            )
        )
    else:
        results.append(pass_result("README", "The main README points to the dataset and validation workflow."))

    if "10.17632/hz2d6gz7pc.4" not in data_readme:
        results.append(fail_result("data/README", "The data README does not mention the reserved DOI."))
    else:
        results.append(pass_result("data/README", "The data README points to the reserved DOI."))

    env_tokens = ["name: spaphish", "python=3.12"]
    missing_env = [token for token in env_tokens if token not in environment]
    if missing_env:
        results.append(
            fail_result(
                "environment.yml",
                f"Missing expected environment tokens: {', '.join(missing_env)}",
            )
        )
    else:
        results.append(pass_result("environment.yml", "The Conda environment is named spaphish and targets Python 3.12."))

    if "beautifulsoup4" not in requirements or "reportlab" not in requirements:
        results.append(fail_result("requirements.txt", "Some expected runtime dependencies are missing."))
    else:
        results.append(pass_result("requirements.txt", "Core runtime dependencies are declared."))

    cells = notebook.get("cells", [])
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "code"
    )
    if "processing.spaphish_processing" in code_source and "main()" in code_source:
        results.append(pass_result("notebook", "The notebook is a lightweight launcher for the processing module."))
    else:
        results.append(fail_result("notebook", "The notebook does not look like the expected launcher."))

    return results


def check_no_llm_references(root: Path) -> list[CheckResult]:
    """Ensure project source and documentation do not mention LLM services."""

    results: list[CheckResult] = []
    banned_patterns = [
        re.compile(r"\bgemini\b", re.IGNORECASE),
        re.compile(r"\bopenai\b", re.IGNORECASE),
        re.compile(r"\bchatgpt\b", re.IGNORECASE),
        re.compile(r"\bgpt\b", re.IGNORECASE),
        re.compile(r"\bollama\b", re.IGNORECASE),
        re.compile(r"\banthropic\b", re.IGNORECASE),
        re.compile(r"\bclaude\b", re.IGNORECASE),
        re.compile(r"\bgroq\b", re.IGNORECASE),
        re.compile(r"\bmistral\b", re.IGNORECASE),
        re.compile(r"\bcohere\b", re.IGNORECASE),
        re.compile(r"\bbedrock\b", re.IGNORECASE),
        re.compile(r"\bvertex\s+ai\b", re.IGNORECASE),
        re.compile(r"\bgenerative\b", re.IGNORECASE),
        re.compile(r"\bllm\b", re.IGNORECASE),
    ]
    scan_roots = [
        root / "README.md",
        root / "data" / "README.md",
        root / "docs",
        root / "analysis",
        root / "processing",
        root / "reporting",
        root / "scripts",
        root / "tools",
        root / "notebooks",
    ]
    excluded_paths = {root / "scripts" / "verify_project.py"}

    offenders: list[str] = []
    for entry in scan_roots:
        files = [entry] if entry.is_file() else [path for path in entry.rglob("*") if path.is_file()]
        for path in files:
            if path in excluded_paths:
                continue
            if path.suffix.lower() not in {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".ipynb"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                offenders.append(f"{path.relative_to(root)}: could not read file ({exc})")
                continue
            for pattern in banned_patterns:
                if pattern.search(text):
                    offenders.append(f"{path.relative_to(root)}: contains '{pattern.pattern}'")
                    break

    if offenders:
        results.append(
            fail_result(
                "llm references",
                "Project documentation or source files still mention LLM-related terms:\n"
                + "\n".join(offenders),
            )
        )
    else:
        results.append(pass_result("llm references", "No LLM-related terms were found in project source or documentation."))

    return results


def check_dataset_files(root: Path) -> list[CheckResult]:
    """Verify that the dataset files expected by the analysis stage are available."""

    results: list[CheckResult] = []
    csv_path = root / "data" / "Spaphish dataset - DiB.csv"
    xlsx_path = root / "data" / "Spaphish dataset - DiB.xlsx"
    if csv_path.exists():
        results.append(pass_result("dataset CSV", f"Found {csv_path.relative_to(root)}."))
    else:
        results.append(fail_result("dataset CSV", f"Missing {csv_path.relative_to(root)}."))

    if xlsx_path.exists():
        results.append(pass_result("dataset XLSX", f"Found {xlsx_path.relative_to(root)}."))
    else:
        results.append(warn_result("dataset XLSX", "The Excel version is absent; CSV is still sufficient."))

    return results


def check_smoke_help(root: Path, timeout: float) -> list[CheckResult]:
    """Run inexpensive CLI checks that should finish quickly."""

    results: list[CheckResult] = []
    help_commands = [
        [sys.executable, str(root / "scripts" / "spaphish.py"), "--help"],
        [sys.executable, str(root / "scripts" / "report_pdf.py"), "--help"],
    ]
    for command in help_commands:
        results.append(run_subprocess(command, root, timeout))
    return results


def check_full_execution(root: Path, timeout: float) -> list[CheckResult]:
    """Run the heavy end-to-end smoke tests for the main pipeline."""

    results: list[CheckResult] = []

    analysis_command = [sys.executable, str(root / "scripts" / "analyze_dataset.py")]
    results.append(run_subprocess(analysis_command, root, timeout))

    html_command = [sys.executable, str(root / "scripts" / "report_html.py")]
    results.append(run_subprocess(html_command, root, timeout))

    pdf_command = [sys.executable, str(root / "scripts" / "report_pdf.py")]
    results.append(run_subprocess(pdf_command, root, timeout))

    raw_dir = root / "data" / "raw"
    raw_eml = list(raw_dir.rglob("*.eml")) if raw_dir.exists() else []
    if raw_eml:
        processing_command = [
            sys.executable,
            str(root / "scripts" / "spaphish.py"),
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(root / "output" / "_verify_processed"),
            "--output-dir",
            str(root / "output" / "_verify_processing"),
            "--no-ocr",
        ]
        results.append(run_subprocess(processing_command, root, timeout))
    else:
        results.append(
            warn_result(
                "processing pipeline",
                "Skipped because data/raw/ does not contain .eml files in this workspace.",
            )
        )

    return results


def print_results(results: Iterable[CheckResult]) -> int:
    """Print validation results and return the number of failures."""

    failures = 0
    warnings = 0
    passes = 0
    for result in results:
        print(f"[{result.status}] {result.name}")
        print(result.details)
        print()
        if result.status == "FAIL":
            failures += 1
        elif result.status == "WARN":
            warnings += 1
        else:
            passes += 1

    print(f"Summary: {passes} passed, {warnings} warnings, {failures} failed")
    return failures


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the validator."""

    parser = argparse.ArgumentParser(description="Validate the SpaPhish repository.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the heavier end-to-end analysis and reporting checks.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Timeout in seconds for each subprocess-based check.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate the project and return a shell exit code."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    root = project_root()

    results: list[CheckResult] = []
    results.extend(check_project_layout(root))
    results.extend(check_python_syntax(root))
    results.extend(check_imports(root))
    results.extend(check_documentation(root))
    results.extend(check_no_llm_references(root))
    results.extend(check_dataset_files(root))
    results.extend(check_smoke_help(root, args.timeout))

    if args.full:
        results.extend(check_full_execution(root, args.timeout))

    failures = print_results(results)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
