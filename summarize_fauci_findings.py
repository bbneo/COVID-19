#!/usr/bin/env python3
"""
Summarize Reading Room PDFs with a locally running LLM (Ollama).

Focuses on findings about Dr. Anthony Fauci: his state of mind, actions, and
interactions with other public health and government officials.

Prerequisites (on your Linux machine):
  1. Download the PDFs first:
       python3 download_paul_reading_room.py
  2. Install Ollama: https://ollama.com
       ollama pull llama3.2
       # or: ollama pull mistral / llama3.1 / qwen2.5 / etc.
  3. Install PDF text extraction:
       pip install -r requirements-summarize.txt
       # optional fallback: sudo apt install poppler-utils

Usage:
  python3 summarize_fauci_findings.py --dry-run
  python3 summarize_fauci_findings.py
  python3 summarize_fauci_findings.py --model llama3.2 --outdir fauci_summaries
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_DOCS_DIR = Path("paul_reading_room_documents")
DEFAULT_OUTDIR = Path("fauci_summaries")
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_CHUNK_CHARS = 12000
DEFAULT_CHUNK_OVERLAP = 800
DEFAULT_MAX_CHUNKS_PER_DOC = 24

FAUCI_KEYWORDS = (
    "fauci",
    "anthony fauci",
    "niaid",
    "nih",
    "collins",
    "francis collins",
    "redfield",
    "robert redfield",
    "birx",
    "deborah birx",
    "daszak",
    "peter daszak",
    "ecohealth",
    "baric",
    "ralph baric",
    "anderssen",  # common OCR/typo near Andersen
    "andersen",
    "kristian andersen",
    "proximal origin",
    "gain-of-function",
    "gain of function",
    "wuhan",
    "lab leak",
    "furin",
    "white house",
    "cdc",
    "fda",
    "hhs",
    "darpa",
    "cia",
    "fbi",
    "odni",
)

SYSTEM_PROMPT = """\
You are a careful research analyst summarizing primary-source documents.
Stick to what the provided text supports. Distinguish clearly between:
- statements attributed to Fauci
- statements about Fauci by others
- inferences you are making from the text
If the excerpt has little or no Fauci-related content, say so briefly.
Do not invent quotations, dates, or participants.
Be concise and concrete."""

CHUNK_PROMPT = """\
Analyze this excerpt from "{doc_name}" (chunk {chunk_num}/{chunk_total}).

Extract and summarize ONLY material relevant to Dr. Anthony Fauci, especially:
1. Fauci's state of mind (private doubts, confidence, frustration, defensiveness, urgency, etc.)
2. Fauci's actions (decisions, directions, public statements, funding choices, communications)
3. Fauci's interactions with other public-health or government officials \
(who, when, about what, and what each side said or did)

Also note any important related officials (e.g. Collins, Redfield, Birx, Daszak, Baric, \
Andersen, intelligence/law-enforcement contacts) when they connect to Fauci.

Use short bullet points. Quote brief key phrases when useful. If nothing relevant appears, \
reply with exactly: NO FAUCI-RELATED CONTENT

Excerpt:
---
{chunk_text}
---"""

DOC_PROMPT = """\
Below are chunk-level notes from the document "{doc_name}".
Synthesize one coherent summary of the key findings about Dr. Anthony Fauci.

Organize under these headings:
## State of mind
## Actions
## Interactions with officials
## Other notable Fauci-related findings
## Uncertainties / gaps

Keep it faithful to the notes. Prefer specific claims with dates/names when available.
If a heading has no support, write "None found in this document."

Chunk notes:
---
{chunk_notes}
---"""

OVERALL_PROMPT = """\
Below are per-document summaries drawn from Sen. Rand Paul's Reading Room releases.
Write an overall briefing on Dr. Anthony Fauci across these documents.

Organize under:
## Cross-cutting themes
## State of mind over time
## Key actions
## Key interactions with public-health and government officials
## Highest-signal documents
## Caveats

Be precise. Do not add facts that are not in the summaries.

Per-document summaries:
---
{doc_summaries}
---"""


@dataclass
class DocResult:
    path: str
    name: str
    pages: int | None
    chars: int
    chunks_total: int
    chunks_used: int
    relevant_chunks: int
    summary: str
    chunk_notes: list[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize downloaded Reading Room PDFs with a local Ollama model, "
            "focusing on Fauci findings."
        )
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help=f"Directory containing downloaded PDFs (default: {DEFAULT_DOCS_DIR})",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Where to write summaries (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_HOST})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=DEFAULT_CHUNK_CHARS,
        help=f"Approximate characters per chunk (default: {DEFAULT_CHUNK_CHARS})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Character overlap between chunks (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS_PER_DOC,
        help=(
            "Max chunks to send to the model per document after relevance ranking "
            f"(default: {DEFAULT_MAX_CHUNKS_PER_DOC})"
        ),
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Optional specific PDF filenames or paths to summarize",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Ollama temperature (default: 0.2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract/chunk text and print plan without calling the LLM",
    )
    parser.add_argument(
        "--skip-overall",
        action="store_true",
        help="Skip the final cross-document briefing",
    )
    return parser.parse_args(argv)


def list_pdfs(docs_dir: Path, files: list[str] | None) -> list[Path]:
    if files:
        paths: list[Path] = []
        for item in files:
            path = Path(item)
            if not path.is_file():
                path = docs_dir / item
            if not path.is_file():
                raise FileNotFoundError(f"PDF not found: {item}")
            paths.append(path)
        return paths

    if not docs_dir.is_dir():
        raise FileNotFoundError(
            f"Documents directory not found: {docs_dir}\n"
            "Run: python3 download_paul_reading_room.py"
        )
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs in {docs_dir}. Run: python3 download_paul_reading_room.py"
        )
    return pdfs


def extract_text_pypdf(path: Path) -> tuple[str, int]:
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts), len(reader.pages)


def extract_text_pdftotext(path: Path) -> tuple[str, int | None]:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout, None


def extract_text(path: Path) -> tuple[str, int | None]:
    try:
        return extract_text_pypdf(path)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"  ! pypdf failed on {path.name}: {exc}", file=sys.stderr)

    if shutil.which("pdftotext"):
        try:
            return extract_text_pdftotext(path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"pdftotext failed for {path}") from exc

    raise RuntimeError(
        "No PDF text extractor available. Install one of:\n"
        "  pip install pypdf\n"
        "  sudo apt install poppler-utils"
    )


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)
    overlap = max(0, min(overlap, chunk_chars // 2))
    while start < length:
        end = min(start + chunk_chars, length)
        if end < length:
            # Prefer breaking on a paragraph/sentence boundary near the end.
            window = text[start:end]
            break_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if break_at >= chunk_chars // 2:
                end = start + break_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def relevance_score(text: str) -> int:
    lowered = text.lower()
    score = 0
    for keyword in FAUCI_KEYWORDS:
        count = lowered.count(keyword)
        if count:
            weight = 5 if "fauci" in keyword else 1
            score += count * weight
    return score


def rank_chunks(chunks: list[str], max_chunks: int) -> list[tuple[int, str]]:
    ranked = sorted(
        ((idx, chunk, relevance_score(chunk)) for idx, chunk in enumerate(chunks)),
        key=lambda item: (-item[2], item[0]),
    )
    # Always keep at least the opening chunk for context if nothing scores.
    selected = [item for item in ranked if item[2] > 0][:max_chunks]
    if not selected and ranked:
        selected = ranked[: min(3, len(ranked))]
    selected_sorted = sorted(selected, key=lambda item: item[0])
    return [(idx, chunk) for idx, chunk, _score in selected_sorted]


def ollama_request(
    host: str,
    path: str,
    payload: dict,
    *,
    timeout: float = 600.0,
) -> dict:
    url = host.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Is it running?\n"
            f"  ollama serve\n"
            f"  ollama pull {payload.get('model', DEFAULT_MODEL)}\n"
            f"Underlying error: {exc}"
        ) from exc


def ollama_list_models(host: str) -> list[str]:
    url = host.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Start it with: ollama serve\n"
            f"Underlying error: {exc}"
        ) from exc
    return [item.get("name", "") for item in payload.get("models", [])]


def ensure_model(host: str, model: str) -> None:
    models = ollama_list_models(host)
    names = set(models)
    # Ollama may list "llama3.2:latest" while user passes "llama3.2"
    if model in names or any(name.split(":")[0] == model for name in names):
        return
    print(f"Model {model!r} not found locally. Available: {models or '(none)'}")
    print(f"Pulling {model!r} via Ollama...")
    subprocess.run(["ollama", "pull", model], check=True)


def chat(
    host: str,
    model: str,
    user_prompt: str,
    *,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    result = ollama_request(host, "/api/chat", payload)
    message = result.get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError(f"Empty response from model {model}")
    return content


def summarize_document(
    path: Path,
    *,
    host: str,
    model: str,
    chunk_chars: int,
    overlap: int,
    max_chunks: int,
    temperature: float,
    dry_run: bool,
) -> DocResult:
    print(f"\n=== {path.name} ===")
    text, pages = extract_text(path)
    text = normalize_text(text)
    chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
    selected = rank_chunks(chunks, max_chunks=max_chunks)
    print(
        f"  extracted {len(text):,} chars"
        + (f" across {pages} pages" if pages is not None else "")
        + f"; {len(chunks)} chunks; using {len(selected)}"
    )

    if dry_run:
        for idx, chunk in selected:
            print(
                f"  - chunk {idx + 1}/{len(chunks)} "
                f"({len(chunk):,} chars, score={relevance_score(chunk)})"
            )
        preview = (
            "DRY RUN — no LLM call. "
            f"Would summarize {len(selected)} chunk(s) from {path.name}."
        )
        return DocResult(
            path=str(path),
            name=path.name,
            pages=pages,
            chars=len(text),
            chunks_total=len(chunks),
            chunks_used=len(selected),
            relevant_chunks=sum(1 for _, chunk in selected if relevance_score(chunk) > 0),
            summary=preview,
            chunk_notes=[],
        )

    chunk_notes: list[str] = []
    relevant = 0
    for order, (idx, chunk) in enumerate(selected, start=1):
        score = relevance_score(chunk)
        print(f"  [llm] chunk {idx + 1}/{len(chunks)} ({order}/{len(selected)}, score={score})")
        note = chat(
            host,
            model,
            CHUNK_PROMPT.format(
                doc_name=path.name,
                chunk_num=idx + 1,
                chunk_total=len(chunks),
                chunk_text=chunk,
            ),
            temperature=temperature,
        )
        if "NO FAUCI-RELATED CONTENT" not in note.upper():
            relevant += 1
            chunk_notes.append(f"[chunk {idx + 1}]\n{note}")
        else:
            chunk_notes.append(f"[chunk {idx + 1}]\nNO FAUCI-RELATED CONTENT")

    if not any("NO FAUCI-RELATED CONTENT" not in note for note in chunk_notes):
        summary = (
            "## State of mind\nNone found in this document.\n\n"
            "## Actions\nNone found in this document.\n\n"
            "## Interactions with officials\nNone found in this document.\n\n"
            "## Other notable Fauci-related findings\nNone found in this document.\n\n"
            "## Uncertainties / gaps\nNo Fauci-related content detected in selected chunks."
        )
    else:
        print("  [llm] synthesizing document summary")
        summary = chat(
            host,
            model,
            DOC_PROMPT.format(
                doc_name=path.name,
                chunk_notes="\n\n".join(chunk_notes),
            ),
            temperature=temperature,
        )

    return DocResult(
        path=str(path),
        name=path.name,
        pages=pages,
        chars=len(text),
        chunks_total=len(chunks),
        chunks_used=len(selected),
        relevant_chunks=relevant,
        summary=summary,
        chunk_notes=chunk_notes,
    )


def write_outputs(outdir: Path, results: list[DocResult], overall: str | None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    docs_dir = outdir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        stem = Path(result.name).stem
        md_path = docs_dir / f"{stem}.md"
        json_path = docs_dir / f"{stem}.json"
        md_path.write_text(
            f"# {result.name}\n\n"
            f"- Characters: {result.chars:,}\n"
            f"- Pages: {result.pages if result.pages is not None else 'unknown'}\n"
            f"- Chunks used: {result.chunks_used}/{result.chunks_total}\n"
            f"- Relevant chunks: {result.relevant_chunks}\n\n"
            f"{result.summary.strip()}\n",
            encoding="utf-8",
        )
        json_path.write_text(
            json.dumps(asdict(result), indent=2),
            encoding="utf-8",
        )

    combined_parts = [
        "# Fauci Findings Summaries",
        "",
        "Generated from Sen. Rand Paul Reading Room documents using a local LLM.",
        "",
    ]
    if overall:
        combined_parts.extend(["## Overall briefing", "", overall.strip(), ""])
    for result in results:
        combined_parts.extend(
            [
                f"## {result.name}",
                "",
                result.summary.strip(),
                "",
            ]
        )
    (outdir / "fauci_findings.md").write_text("\n".join(combined_parts), encoding="utf-8")
    if overall:
        (outdir / "overall_briefing.md").write_text(overall.strip() + "\n", encoding="utf-8")
    (outdir / "manifest.json").write_text(
        json.dumps(
            {
                "documents": [asdict(result) for result in results],
                "overall": overall,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pdfs = list_pdfs(args.docs_dir, args.files)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Found {len(pdfs)} PDF(s) in {args.docs_dir}")
    if not args.dry_run:
        if shutil.which("ollama") is None:
            print(
                "Warning: `ollama` CLI not found on PATH. "
                "Will still try the HTTP API at "
                f"{args.host}",
                file=sys.stderr,
            )
        try:
            ensure_model(args.host, args.model)
        except Exception as exc:  # noqa: BLE001
            # ensure_model may fail if CLI missing but HTTP works with model already pulled
            try:
                models = ollama_list_models(args.host)
            except Exception:
                print(str(exc), file=sys.stderr)
                return 1
            if not (
                args.model in models
                or any(name.split(":")[0] == args.model for name in models)
            ):
                print(
                    f"Model {args.model!r} is not available at {args.host}.\n"
                    f"Available: {models or '(none)'}\n"
                    f"Install/pull with: ollama pull {args.model}",
                    file=sys.stderr,
                )
                return 1

    results: list[DocResult] = []
    started = time.time()
    for pdf in pdfs:
        try:
            result = summarize_document(
                pdf,
                host=args.host,
                model=args.model,
                chunk_chars=max(args.chunk_chars, 2000),
                overlap=max(args.overlap, 0),
                max_chunks=max(args.max_chunks, 1),
                temperature=args.temperature,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed on {pdf.name}: {exc}", file=sys.stderr)
            return 1
        results.append(result)

    overall = None
    if not args.dry_run and not args.skip_overall and results:
        print("\n[llm] writing overall briefing")
        doc_summaries = "\n\n".join(
            f"### {result.name}\n{result.summary}" for result in results
        )
        overall = chat(
            args.host,
            args.model,
            OVERALL_PROMPT.format(doc_summaries=doc_summaries),
            temperature=args.temperature,
        )

    write_outputs(args.outdir, results, overall)
    elapsed = time.time() - started
    print(f"\nWrote summaries to {args.outdir.resolve()}")
    print(f"Main report: {args.outdir.resolve() / 'fauci_findings.md'}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
