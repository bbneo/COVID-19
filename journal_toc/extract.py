"""Pull text out of table-of-contents PDFs."""

from __future__ import annotations

import warnings
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfReadWarning

from journal_toc.errors import TocError

warnings.filterwarnings("ignore", category=PdfReadWarning)


class ExtractionError(TocError):
    """The PDF could not be read as a table of contents."""


def extract_pages(path: Path, *, max_pages: int) -> list[str]:
    """Return one string per page, in order, up to ``max_pages``."""
    if not path.is_file():
        raise ExtractionError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ExtractionError(f"Not a PDF file: {path}")
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # pragma: no cover - pypdf raises various types
                raise ExtractionError(
                    f"{path.name} is encrypted. Decrypt it, then run the summarizer again."
                ) from exc
        total = len(reader.pages)
        count = min(total, max_pages)
        pages: list[str] = []
        for index in range(count):
            raw = reader.pages[index].extract_text() or ""
            pages.append(raw.replace("\x00", " ").strip())
    except ExtractionError:
        raise
    except PdfReadError as exc:
        raise ExtractionError(f"Could not read {path.name}: {exc}") from exc
    except Exception as exc:
        raise ExtractionError(f"Could not read {path.name}: {exc}") from exc
    if total > max_pages:
        pages.append(
            f"[Stopped after {max_pages} of {total} pages. "
            "Pass --max-pages to read further.]"
        )
    return pages


def strip_repeated_lines(pages: list[str]) -> list[str]:
    """Drop running headers and footers that show up on most TOC pages.

    The first page keeps those lines so an issue date printed only in the
    banner is still visible to the model.
    """
    content_pages = [page for page in pages if not page.startswith("[Stopped after")]
    notes = [page for page in pages if page.startswith("[Stopped after")]
    if len(content_pages) < 3:
        return pages
    split_pages = [[line.strip() for line in page.splitlines()] for page in content_pages]
    counts: dict[str, int] = {}
    for lines in split_pages:
        for line in set(lines):
            if line:
                counts[line] = counts.get(line, 0) + 1
    threshold = max(3, (len(content_pages) + 1) // 2)
    repeated = {line for line, count in counts.items() if count >= threshold and len(line) <= 140}
    cleaned: list[str] = []
    for index, lines in enumerate(split_pages):
        if index == 0:
            kept = lines
        else:
            kept = [line for line in lines if line not in repeated]
        cleaned.append("\n".join(line for line in kept if line))
    return cleaned + notes


def page_text(pages: list[str]) -> str:
    """Join pages with a form feed so the model can see page breaks."""
    return "\n\f\n".join(page for page in pages if page.strip())


def has_readable_text(text: str) -> bool:
    letters = sum(1 for char in text if char.isalpha())
    return letters >= 40


def chunk_text(text: str, *, max_chars: int) -> list[str]:
    """Split TOC text on page breaks so each model call stays within context."""
    if len(text) <= max_chars:
        return [text]
    pages = text.split("\n\f\n")
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for page in pages:
        piece = page.strip()
        if not piece:
            continue
        extra = len(piece) + (3 if current else 0)
        if current and size + extra > max_chars:
            chunks.append("\n\f\n".join(current))
            current = [piece]
            size = len(piece)
        else:
            current.append(piece)
            size += extra
    if current:
        chunks.append("\n\f\n".join(current))
    # A single page can still exceed max_chars. Hard-split it.
    bounded: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            bounded.append(chunk)
            continue
        start = 0
        while start < len(chunk):
            bounded.append(chunk[start : start + max_chars])
            start += max_chars
    return bounded
