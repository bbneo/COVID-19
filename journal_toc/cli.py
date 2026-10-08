"""Command line for the journal table-of-contents summarizer."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from journal_toc import __version__
from journal_toc.dates import coerce_month, coerce_year
from journal_toc.errors import TocError
from journal_toc.llm import LlmClient
from journal_toc.render import render_file_markdown, write_model_error, write_summaries
from journal_toc.summarize import FileSummary, summarize_pdf

# The folder on disk is JAMA_2019-2020. The hyphenated name is accepted too.
SAMPLE_DIRECTORIES = (
    Path("~/Dropbox/PublicHealth/Covid-2026/JAMA_2019-2020"),
    Path("~/Dropbox/PublicHealth/Covid-2026/JAMA-2019-2020"),
)
logger = logging.getLogger("journal_toc")


def default_sample_dir() -> Path:
    """Return the JAMA sample folder, preferring the one that exists."""
    for candidate in SAMPLE_DIRECTORIES:
        if candidate.expanduser().is_dir():
            return candidate
    return SAMPLE_DIRECTORIES[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summarize_journal_toc",
        description=(
            "Summarize medical-journal table-of-contents PDFs with a local LLM. "
            "Each file is catalogued by topic and article type, with the issue year and month."
        ),
        epilog=(
            "Sample JAMA 2019-2020 contents pages:\n"
            "  ~/Dropbox/PublicHealth/Covid-2026/JAMA_2019-2020\n"
            "\n"
            "From that folder, after Ollama is running:\n"
            "  python /path/to/COVID-19/scripts/summarize_journal_toc.py .\n"
            "\n"
            "Start a local model first, for example:\n"
            "  ollama pull llama3.2\n"
            "  ollama serve"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "PDF files or directories of table-of-contents PDFs. "
            "When omitted, the JAMA 2019-2020 sample directory above is used."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("toc-summaries"),
        help="Directory for Markdown, JSON, and articles.csv (default: ./toc-summaries).",
    )
    parser.add_argument("--model", default="llama3.2", help="Local model name (default: llama3.2).")
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Local model server (default: Ollama at http://127.0.0.1:11434).",
    )
    parser.add_argument(
        "--backend",
        choices=("ollama", "openai"),
        default="ollama",
        help="ollama uses /api/chat. openai uses an OpenAI-compatible /v1/chat/completions server.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for each model call (default: 900).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=4096,
        help="Ollama context length. Lower this if the model is slow to load.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=200,
        help="Most pages to read from each PDF (default: 200).",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=3500,
        help="Extracted characters sent in one model call (default: 3500).",
    )
    parser.add_argument("--year", type=int, help="Issue year, if you want to set or filter it.")
    parser.add_argument("--month", help="Issue month as a number (1-12) or name such as March.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include PDFs in subdirectories.",
    )
    parser.add_argument(
        "--save-text",
        action="store_true",
        help="Also write the extracted PDF text next to each summary.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print each Markdown summary to standard output as well as saving it.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def collect_pdfs(paths: list[str], *, recursive: bool) -> list[Path]:
    """Resolve files and directories to a sorted list of PDFs."""
    chosen = paths or [str(default_sample_dir())]
    found: list[Path] = []
    missing: list[str] = []
    for raw in chosen:
        path = Path(raw).expanduser()
        if not path.exists():
            missing.append(str(path))
            continue
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            found.extend(
                item
                for item in path.glob(pattern)
                if item.is_file() and item.suffix.lower() == ".pdf"
            )
        elif path.is_file() and path.suffix.lower() == ".pdf":
            found.append(path)
        else:
            missing.append(f"{path} (not a PDF)")
    if missing and not found:
        joined = "\n  ".join(missing)
        hint = ""
        if not paths:
            hint = (
                "\nPass PDF files or a directory. The sample pages are at "
                "~/Dropbox/PublicHealth/Covid-2026/JAMA_2019-2020."
            )
        raise TocError(f"No table-of-contents PDFs found:\n  {joined}{hint}")
    if missing:
        for item in missing:
            logger.warning("Skipping %s", item)
    unique = prefer_ocr_copies(_dedupe(found))
    if not unique:
        raise TocError("No PDF files found in the given paths.")
    return sorted(unique, key=lambda item: (item.name.casefold(), str(item)))


def prefer_ocr_copies(paths: list[Path]) -> list[Path]:
    """Drop a scan when the same folder already has its ``_ocr`` copy.

    ``Selected_JAMA_Contents_2019.pdf`` is skipped when
    ``Selected_JAMA_Contents_2019_ocr.pdf`` is present, because the scan has
    no text layer and the OCR file is the one the model can read.
    """
    ocr_stems = {
        path.stem.casefold()
        for path in paths
        if path.stem.casefold().endswith("_ocr")
    }
    kept: list[Path] = []
    for path in paths:
        ocr_stem = f"{path.stem}_ocr".casefold()
        if ocr_stem in ocr_stems:
            logger.info(
                "Skipping %s because %s_ocr%s has the text layer.",
                path.name,
                path.stem,
                path.suffix,
            )
            continue
        kept.append(path)
    return kept


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        month = _parse_month_flag(args.month)
        year = _parse_year_flag(args.year)
        if (year is None) != (month is None):
            raise TocError("Pass --year and --month together.")
        pdfs = collect_pdfs(args.paths, recursive=args.recursive)
    except TocError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    client = LlmClient(
        host=args.host,
        model=args.model,
        backend=args.backend,
        timeout=args.timeout,
        num_ctx=args.num_ctx,
    )
    logger.info("Checking that %s answers at %s", args.model, args.host)
    probe = LlmClient(
        host=args.host,
        model=args.model,
        backend=args.backend,
        timeout=min(90.0, args.timeout),
        num_ctx=args.num_ctx,
    )
    try:
        probe.complete("Reply with a JSON object and no other text.", '{"ok": true}')
    except TocError as exc:
        print(f"The local model did not answer a short test prompt. {exc}", file=sys.stderr)
        print("In another terminal, run: ollama run llama3.2", file=sys.stderr)
        return 1
    logger.info("Local model answered.")
    summaries: list[FileSummary] = []
    failures = 0
    for index, path in enumerate(pdfs, start=1):
        logger.info("[%s/%s] %s", index, len(pdfs), path)
        try:
            summary = summarize_pdf(
                path,
                client,
                year=year,
                month=month,
                max_pages=args.max_pages,
                chunk_chars=args.chunk_chars,
            )
        except TocError as exc:
            failures += 1
            print(f"{path.name}: {exc}", file=sys.stderr)
            raw = getattr(exc, "raw", "")
            saved = write_model_error(args.output, path, raw)
            if saved is not None:
                print(f"Model response written to {saved}", file=sys.stderr)
            continue
        summaries.append(summary)
        if args.stdout:
            print(render_file_markdown(summary))
        written = write_summaries(summaries, args.output, save_text=args.save_text)
        logger.info("Saved %s so far in %s", len(summaries), args.output)
    if not summaries:
        print("No summaries were written.", file=sys.stderr)
        return 1
    written = write_summaries(summaries, args.output, save_text=args.save_text)
    logger.info(
        "Wrote %s summaries (%s articles) to %s",
        len(summaries),
        sum(len(issue.articles) for summary in summaries for issue in summary.issues),
        args.output,
    )
    for path in written:
        logger.info("  %s", path)
    return 1 if failures else 0


def _parse_month_flag(value: str | None) -> int | None:
    if value is None:
        return None
    month = coerce_month(value)
    if month is None:
        raise TocError("--month must be 1-12 or a month name such as March.")
    return month


def _parse_year_flag(value: int | None) -> int | None:
    if value is None:
        return None
    year = coerce_year(value)
    if year is None:
        raise TocError("--year must be a four-digit year.")
    return year


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique
