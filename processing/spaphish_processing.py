#!/usr/bin/env python
"""SpaPhish raw email processing pipeline.

This module turns a folder of `.eml` files into a cleaned, structured
dataset. It performs:

* SHA-256 deduplication
* email parsing
* plain text / HTML body extraction
* URL extraction
* attachment accounting
* optional OCR over inline or attached images
* CSV export plus duplicate / error logs

The module is intentionally standalone so the notebook can simply call
`main()` instead of duplicating the implementation.
"""

from __future__ import annotations

import argparse
import base64
import io
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - checked at runtime
    BeautifulSoup = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - checked at runtime
    Image = None

try:
    import pytesseract

    _HAS_TESSERACT = True
except ImportError:  # pragma: no cover - OCR can be disabled
    pytesseract = None
    _HAS_TESSERACT = False


URL_REGEX = re.compile(r"(?i)\b((?:https?://|www\.)[^\s<>'\"\)\]]+)")


@dataclass(frozen=True)
class ProcessingPaths:
    """Resolved paths used by the processing pipeline."""

    project_root: Path
    raw_dir: Path
    processed_dir: Path
    output_dir: Path
    dataset_csv: Path
    duplicates_log: Path
    hashes_log: Path
    processing_errors_log: Path
    ocr_errors_log: Path
    summary_json: Path


def project_root() -> Path:
    """Return the repository root that contains the `processing/` folder."""

    return Path(__file__).resolve().parents[1]


def require_beautifulsoup() -> None:
    """Raise a clear error if BeautifulSoup is unavailable."""

    if BeautifulSoup is None:
        raise RuntimeError(
            "beautifulsoup4 is required. Install dependencies with `pip install -r requirements.txt`."
        )


def require_pillow() -> None:
    """Raise a clear error if Pillow is unavailable."""

    if Image is None:
        raise RuntimeError(
            "Pillow is required. Install dependencies with `pip install -r requirements.txt`."
        )


def resolve_paths(
    root: Path,
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> ProcessingPaths:
    """Resolve the standard SpaPhish processing paths."""

    raw_dir = raw_dir or root / "data" / "raw"
    processed_dir = processed_dir or root / "data" / "processed"
    output_dir = output_dir or root / "output" / "processing"
    return ProcessingPaths(
        project_root=root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        output_dir=output_dir,
        dataset_csv=output_dir / "spaphish_eml_dataset.csv",
        duplicates_log=output_dir / "duplicates.txt",
        hashes_log=output_dir / "hashes.txt",
        processing_errors_log=output_dir / "processing_errors.log",
        ocr_errors_log=output_dir / "ocr_errors.log",
        summary_json=output_dir / "processing_summary.json",
    )


def ensure_directories(paths: ProcessingPaths) -> None:
    """Create the working directories used by the pipeline."""

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.output_dir.mkdir(parents=True, exist_ok=True)


def collect_eml_files(folder: Path) -> list[Path]:
    """Return all `.eml` files found under `folder`."""

    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*.eml") if p.is_file())


def file_hash(path: Path, block_size: int = 65536) -> str:
    """Return the SHA-256 hash of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_email(path: Path):
    """Parse an `.eml` file into an email message object."""

    parser = BytesParser(policy=policy.default)
    return parser.parsebytes(path.read_bytes())


def decode_email_text(part) -> str:
    """Decode a MIME part into text, preserving content when decoding fails."""

    try:
        content = part.get_content()
        return content if isinstance(content, str) else str(content)
    except Exception:
        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")


def encode_list(values: Iterable[str]) -> str:
    """Encode a list of values using the notebook-compatible bracket format."""

    return ",".join(f"[{value}]" for value in values)


def encode_sizes(values: Iterable[int]) -> str:
    """Encode a list of sizes using the notebook-compatible bracket format."""

    return ",".join(f"[{int(value)}]" for value in values)


def build_cid_map(msg) -> dict[str, bytes]:
    """Map Content-ID values to their decoded bytes."""

    cid_map: dict[str, bytes] = {}
    for part in msg.walk():
        cid = part.get("Content-ID")
        if not cid:
            continue
        key = cid.strip().lstrip("<").rstrip(">").strip()
        try:
            payload = part.get_payload(decode=True) or b""
        except Exception:
            payload = b""
        cid_map[key] = payload
    return cid_map


def extract_images_from_html(html_text: Optional[str], msg, allow_remote_images: bool = False) -> list[bytes]:
    """Extract inline image bytes from HTML content."""

    if not html_text:
        return []
    require_beautifulsoup()

    images: list[bytes] = []
    cid_map = build_cid_map(msg)
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup.find_all("img", src=True):
        src = (tag.get("src") or "").strip()
        if not src:
            continue
        src_l = src.lower()
        if src_l.startswith("data:image/"):
            try:
                _, payload = src.split(",", 1)
                images.append(base64.b64decode(payload))
            except Exception:
                continue
        elif src_l.startswith("cid:"):
            cid = src.split(":", 1)[1].strip().lstrip("<").rstrip(">")
            data = cid_map.get(cid)
            if data:
                images.append(data)
        elif src_l.startswith("http://") or src_l.startswith("https://"):
            if allow_remote_images:
                # Remote image downloading is intentionally disabled by default
                # to keep the pipeline deterministic and privacy-friendly.
                continue
    return images


def extract_images_from_parts(msg) -> list[bytes]:
    """Return all bytes from image/* MIME parts."""

    images: list[bytes] = []
    for part in msg.walk():
        if part.get_content_type().startswith("image/"):
            try:
                payload = part.get_payload(decode=True) or b""
            except Exception:
                payload = b""
            if payload:
                images.append(payload)
    return images


def preprocess_image_for_ocr(img_bytes: bytes) -> Image.Image:
    """Load image bytes into a PIL image ready for OCR."""

    require_pillow()
    image = Image.open(io.BytesIO(img_bytes))
    if getattr(image, "is_animated", False):
        image.seek(0)
    return image.convert("L")


def log_line(path: Path, text: str) -> None:
    """Append one text line to a log file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def log_ocr_error(
    ocr_errors_log: Path,
    relative_path: str,
    file_hash_value: str,
    image_index: int,
    err_type: str,
    err_msg: str,
) -> None:
    """Record an OCR error in a structured text line."""

    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    log_line(
        ocr_errors_log,
        f"[{ts}] {relative_path} | hash={file_hash_value} | img_idx={image_index} | "
        f"ERROR={err_type} | msg={err_msg}",
    )


def log_processing_error(
    processing_errors_log: Path,
    relative_path: str,
    file_hash_value: str,
    err_type: str,
    err_msg: str,
) -> None:
    """Record a parsing or extraction error in a structured text line."""

    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    log_line(
        processing_errors_log,
        f"[{ts}] {relative_path} | hash={file_hash_value} | ERROR={err_type} | msg={err_msg}",
    )


def deduplicate_eml_files(
    source_dir: Path,
    processed_dir: Path,
    duplicates_log: Path,
    hashes_log: Path,
) -> tuple[dict[str, int], list[Path]]:
    """Copy unique `.eml` files to `processed_dir` and log duplicates."""

    seen_hashes: dict[str, str] = {}
    copied = 0
    duplicates = 0
    total = 0
    unique_files: list[Path] = []

    duplicates_log.write_text("Duplicate records found:\n\n", encoding="utf-8")
    with hashes_log.open("w", encoding="utf-8") as hashes_handle:
        hashes_handle.write("Hash inventory for each file:\n\n")
        with duplicates_log.open("a", encoding="utf-8") as duplicates_handle:
            for eml_file in collect_eml_files(source_dir):
                total += 1
                file_hash_value = file_hash(eml_file)
                try:
                    relative_file = eml_file.relative_to(source_dir)
                except ValueError:
                    relative_file = Path(eml_file.name)
                hashes_handle.write(f"{relative_file} -> {file_hash_value}\n")
                destination = processed_dir / relative_file
                destination.parent.mkdir(parents=True, exist_ok=True)
                if file_hash_value not in seen_hashes:
                    seen_hashes[file_hash_value] = str(relative_file)
                    shutil.copy2(eml_file, destination)
                    copied += 1
                    unique_files.append(destination)
                else:
                    duplicates += 1
                    duplicates_handle.write(
                        f"Duplicate ignored: {relative_file} (same as {seen_hashes[file_hash_value]})\n"
                    )

    return (
        {
            "total": total,
            "copied": copied,
            "duplicates": duplicates,
            "unique_hashes": len(seen_hashes),
        },
        unique_files,
    )


def parse_email_date(msg) -> str:
    """Return an ISO formatted date string extracted from the email headers."""

    raw_date = msg.get("Date")
    if not raw_date:
        return ""
    try:
        dt = parsedate_to_datetime(raw_date)
        if dt is None:
            return ""
        return dt.isoformat()
    except Exception:
        return ""


def collect_urls(text_plain: Optional[str], text_html: Optional[str]) -> tuple[int, str]:
    """Extract and encode URLs from text and HTML body content."""

    urls: list[str] = []
    for value in (text_plain, text_html):
        if not value:
            continue
        urls.extend(match.group(1) for match in URL_REGEX.finditer(value))
    cleaned = [url.rstrip(".,);'\"!?]") for url in urls if url.strip()]
    unique_urls = list(dict.fromkeys(cleaned))
    return len(unique_urls), encode_list(unique_urls)


def count_hops(msg) -> int:
    """Count the number of `Received` headers."""

    return len(msg.get_all("Received", []))


def extract_bodies_and_media(msg) -> dict[str, object]:
    """Extract plain text, HTML, media flags, and attachment metadata."""

    text_plain = None
    text_html = None
    has_inline_img_tag = False
    has_data_uri_img = False
    has_cid_img = False
    has_img_attach_inline = False
    attachment_types: list[str] = []
    attachment_sizes: list[int] = []
    attachments_total_size = 0
    attachments_count = 0

    for part in msg.walk():
        content_type = part.get_content_type()
        disposition = (part.get_content_disposition() or "").lower()
        filename = part.get_filename()

        if content_type == "text/plain" and text_plain is None:
            text_plain = decode_email_text(part)
        elif content_type == "text/html" and text_html is None:
            text_html = decode_email_text(part)

        if content_type.startswith("image/") and (disposition == "inline" or not filename):
            has_img_attach_inline = True

        if filename:
            attachments_count += 1
            suffix = Path(filename).suffix.lower().lstrip(".")
            attachment_types.append(suffix or content_type.split("/")[-1].lower())
            try:
                payload = part.get_payload(decode=True) or b""
            except Exception:
                payload = b""
            attachment_sizes.append(len(payload))
            attachments_total_size += len(payload)

    if text_html:
        require_beautifulsoup()
        soup = BeautifulSoup(text_html, "html.parser")
        for img in soup.find_all("img", src=True):
            has_inline_img_tag = True
            src = (img.get("src") or "").strip().lower()
            if src.startswith("data:image/"):
                has_data_uri_img = True
            if src.startswith("cid:"):
                has_cid_img = True

    if text_plain is not None:
        body_type = "text/plain"
    elif text_html is not None:
        body_type = "html+image" if (has_inline_img_tag or has_img_attach_inline) else "text/html"
    else:
        body_type = "unknown"

    media_flags: list[str] = []
    if has_inline_img_tag:
        media_flags.append("INLINE_IMG_TAG")
    if has_data_uri_img:
        media_flags.append("DATA_URI_IMG")
    if has_cid_img:
        media_flags.append("CID_IMG")
    if has_img_attach_inline:
        media_flags.append("IMG_ATTACHMENT_INLINE")

    any_image = bool(media_flags)
    return {
        "text_plain": text_plain,
        "text_html": text_html,
        "body_type": body_type,
        "media_flag": "IMAGE" if any_image else "",
        "body_media_flags": encode_list(media_flags),
        "attachments_count": attachments_count,
        "attachments_types": encode_list(attachment_types),
        "attachments_sizes": encode_sizes(attachment_sizes),
        "attachments_total_size": int(attachments_total_size),
    }


def extract_text_excerpt(text: str, max_len: int = 400) -> str:
    """Build a short text excerpt for reports or examples."""

    clean = " ".join((text or "").split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3].rstrip() + "..."


def build_dataset_record(
    eml_path: Path,
    source_root: Path,
    enable_ocr: bool,
    ocr_lang: str,
    max_images_per_email: int,
    max_ocr_text_chars: int,
    allow_remote_images: bool,
    ocr_errors_log: Path,
) -> dict[str, object]:
    """Parse one email and return a row for the output dataset."""

    message = read_email(eml_path)
    relative_path = str(eml_path.relative_to(source_root)) if source_root in eml_path.parents else str(eml_path.name)
    current_hash = file_hash(eml_path)
    meta = extract_bodies_and_media(message)
    url_count, urls_encoded = collect_urls(meta["text_plain"], meta["text_html"])
    hops_count = count_hops(message)

    ocr_text = ""
    ocr_image_count = 0
    if enable_ocr:
        images = []
        images.extend(extract_images_from_html(meta["text_html"], message, allow_remote_images=allow_remote_images))
        images.extend(extract_images_from_parts(message))
        if max_images_per_email is not None:
            images = images[:max_images_per_email]
        if images and _HAS_TESSERACT:
            texts = []
            for index, image_bytes in enumerate(images, start=1):
                try:
                    image = preprocess_image_for_ocr(image_bytes)
                    text = pytesseract.image_to_string(image, lang=ocr_lang) or ""
                    if text.strip():
                        texts.append(text.strip())
                        ocr_image_count += 1
                except Exception as exc:
                    log_ocr_error(
                        ocr_errors_log,
                        relative_path,
                        current_hash,
                        index,
                        type(exc).__name__,
                        str(exc).replace("\n", " ")[:1000],
                    )
            ocr_text = "\n\n".join(texts)
            if len(ocr_text) > max_ocr_text_chars:
                ocr_text = ocr_text[:max_ocr_text_chars] + " ...[TRUNCATED]"

    body_raw = meta["text_plain"] if meta["text_plain"] is not None else (meta["text_html"] or "")
    body_text = meta["text_plain"] if meta["text_plain"] is not None else ""
    if not body_text and meta["text_html"]:
        require_beautifulsoup()
        body_text = BeautifulSoup(meta["text_html"], "html.parser").get_text(separator=" ", strip=True)

    return {
        "relative_path": relative_path,
        "hash": current_hash,
        "subject": message.get("Subject") or "",
        "date": parse_email_date(message),
        "body_raw": body_raw,
        "body_text": body_text,
        "body_type": meta["body_type"],
        "media_flag": meta["media_flag"],
        "body_media_flags": meta["body_media_flags"],
        "url_count": url_count,
        "urls": urls_encoded,
        "attachments_count": meta["attachments_count"],
        "attachments_types": meta["attachments_types"],
        "attachments_total_size": meta["attachments_total_size"],
        "attachments_sizes": meta["attachments_sizes"],
        "hops_count": hops_count,
        "ocr_image_count": ocr_image_count,
        "ocr_text": ocr_text,
    }


def process_dataset(
    paths: ProcessingPaths,
    enable_ocr: bool = True,
    ocr_lang: str = "eng+spa",
    max_images_per_email: int = 20,
    max_ocr_text_chars: int = 20000,
    allow_remote_images: bool = False,
) -> pd.DataFrame:
    """Run the full processing pipeline and return the dataset DataFrame."""

    ensure_directories(paths)

    if not paths.raw_dir.exists():
        raise FileNotFoundError(
            f"Raw email folder not found: {paths.raw_dir}. Place the `.eml` files there first."
        )

    if not _HAS_TESSERACT:
        enable_ocr = False

    source_files = collect_eml_files(paths.raw_dir)
    if not source_files:
        fallback_files = collect_eml_files(paths.processed_dir)
        if not fallback_files:
            raise FileNotFoundError(
                f"No `.eml` files found in {paths.raw_dir} or {paths.processed_dir}."
            )
        source_root = paths.processed_dir
        unique_files = fallback_files
        dedup_stats = {"total": len(unique_files), "copied": len(unique_files), "duplicates": 0, "unique_hashes": len(unique_files)}
    else:
        dedup_stats, unique_files = deduplicate_eml_files(
            paths.raw_dir,
            paths.processed_dir,
            paths.duplicates_log,
            paths.hashes_log,
        )
        source_root = paths.processed_dir

    rows: list[dict[str, object]] = []
    processed_ok = 0
    processing_errors = 0
    total_urls = 0
    total_attachments = 0
    total_ocr_images = 0

    paths.processing_errors_log.write_text("Processing errors log:\n\n", encoding="utf-8")
    paths.ocr_errors_log.write_text("OCR error log:\n\n", encoding="utf-8")

    for eml_path in unique_files:
        try:
            record = build_dataset_record(
                eml_path=eml_path,
                source_root=source_root,
                enable_ocr=enable_ocr,
                ocr_lang=ocr_lang,
                max_images_per_email=max_images_per_email,
                max_ocr_text_chars=max_ocr_text_chars,
                allow_remote_images=allow_remote_images,
                ocr_errors_log=paths.ocr_errors_log,
            )
            rows.append(record)
            processed_ok += 1
            total_urls += int(record["url_count"])
            total_attachments += int(record["attachments_count"])
            total_ocr_images += int(record["ocr_image_count"])
        except Exception as exc:
            processing_errors += 1
            relative = str(eml_path.relative_to(source_root)) if source_root in eml_path.parents else str(eml_path.name)
            try:
                file_hash_value = file_hash(eml_path)
            except Exception:
                file_hash_value = "N/A"
            log_processing_error(
                paths.processing_errors_log,
                relative,
                file_hash_value,
                type(exc).__name__,
                str(exc).replace("\n", " ")[:1000],
            )

    df = pd.DataFrame.from_records(rows)
    df.to_csv(paths.dataset_csv, index=False, encoding="utf-8")

    summary = {
        "raw_eml_files": dedup_stats["total"],
        "unique_eml_files": dedup_stats["copied"],
        "duplicate_eml_files": dedup_stats["duplicates"],
        "processed_ok": processed_ok,
        "processing_errors": processing_errors,
        "total_urls": total_urls,
        "total_attachments": total_attachments,
        "total_ocr_images": total_ocr_images,
        "ocr_enabled": bool(enable_ocr and _HAS_TESSERACT),
        "dataset_csv": str(paths.dataset_csv),
    }
    paths.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dataset saved: {len(df)} rows -> {paths.dataset_csv}")
    print(f"Logs: {paths.processing_errors_log} | {paths.ocr_errors_log}")
    print(f"Summary: {paths.summary_json}")
    return df


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="SpaPhish raw email processing pipeline.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Repository root")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Folder with raw `.eml` files")
    parser.add_argument("--processed-dir", type=Path, default=None, help="Folder for deduplicated `.eml` files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Folder for processing outputs")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR even if pytesseract is installed")
    parser.add_argument("--ocr-lang", default="eng+spa", help="Tesseract language code, e.g. eng+spa")
    parser.add_argument("--max-images-per-email", type=int, default=20, help="Maximum images to OCR per email")
    parser.add_argument("--max-ocr-text-chars", type=int, default=20000, help="Maximum OCR text length per email")
    parser.add_argument("--allow-remote-images", action="store_true", help="Allow remote image sources in HTML")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the SpaPhish processing pipeline and return an exit code."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    paths = resolve_paths(
        root=args.project_root,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
    )

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


if __name__ == "__main__":
    raise SystemExit(main())
