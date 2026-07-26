#!/usr/bin/env python3
"""
Download all documents linked from Senator Rand Paul's Reading Room.

Default source:
  https://www.paul.senate.gov/readingroom/?ref=0725

Many "Read the Documents" buttons point at pretty URLs that redirect to PDFs
under /wp-content/uploads/. This script follows those redirects, scrapes any
HTML package pages for nested files, and saves everything locally.

Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

DEFAULT_URL = "https://www.paul.senate.gov/readingroom/?ref=0725"
DEFAULT_OUTDIR = Path("paul_reading_room_documents")
USER_AGENT = (
    "Mozilla/5.0 (compatible; PaulReadingRoomDownloader/1.0; "
    "+https://www.paul.senate.gov/readingroom/)"
)

DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".csv",
    ".txt",
    ".rtf",
    ".odt",
    ".ods",
}

SKIP_PATH_FRAGMENTS = (
    "/wp-content/plugins/",
    "/wp-includes/",
    "/wp-json/",
    "/feed/",
    "/xmlrpc.php",
)

SKIP_EXTENSIONS = {
    ".css",
    ".js",
    ".map",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
}


@dataclass(frozen=True)
class Document:
    source_url: str
    final_url: str
    link_text: str
    content_type: str
    content_length: int | None
    filename: str


class LinkParser(HTMLParser):
    """Collect anchor href/text pairs from an HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._in_a = False
        self._href: str | None = None
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        href = attr_map.get("href", "").strip()
        if not href:
            return
        self._in_a = True
        self._href = href
        self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._in_a:
            self._chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._in_a:
            return
        text = html_lib.unescape(re.sub(r"\s+", " ", "".join(self._chunks))).strip()
        assert self._href is not None
        self.links.append((self._href, text))
        self._in_a = False
        self._href = None
        self._chunks = []


def build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.HTTPRedirectHandler())


def request(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
    read_body: bool = True,
) -> tuple[urllib.request.addinfourl, bytes | None]:
    req_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers, method=method)
    response = opener.open(req, timeout=timeout)
    if method.upper() == "HEAD" or not read_body:
        return response, None
    return response, response.read()


def normalize_url(base: str, href: str) -> str | None:
    href = html_lib.unescape(href).strip()
    if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
        return None
    return urllib.parse.urljoin(base, href)


def path_extension(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    return Path(path).suffix.lower()


def looks_like_asset(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    if any(fragment in path for fragment in SKIP_PATH_FRAGMENTS):
        return True
    if path_extension(url) in SKIP_EXTENSIONS:
        return True
    if parsed.scheme not in {"http", "https"}:
        return True
    return False


def looks_like_document_url(url: str) -> bool:
    return path_extension(url) in DOCUMENT_EXTENSIONS


def is_same_site(url: str, root: str) -> bool:
    return urllib.parse.urlparse(url).netloc == urllib.parse.urlparse(root).netloc


def sanitize_filename(name: str) -> str:
    name = urllib.parse.unquote(name).strip().replace("\x00", "")
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name or "document"


def filename_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    name = Path(path).name or "document"
    return sanitize_filename(name)


def filename_from_content_disposition(header: str | None) -> str | None:
    if not header:
        return None
    # RFC 5987: filename*=UTF-8''...
    match = re.search(r"filename\*\s*=\s*([^']+)''([^;]+)", header, flags=re.I)
    if match:
        return sanitize_filename(urllib.parse.unquote(match.group(2).strip().strip('"')))
    match = re.search(r'filename\s*=\s*"([^"]+)"', header, flags=re.I)
    if match:
        return sanitize_filename(match.group(1))
    match = re.search(r"filename\s*=\s*([^;]+)", header, flags=re.I)
    if match:
        return sanitize_filename(match.group(1).strip().strip('"'))
    return None


def unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:8]
    alt = directory / f"{stem}-{digest}{suffix}"
    if not alt.exists():
        return alt
    index = 2
    while True:
        alt = directory / f"{stem}-{digest}-{index}{suffix}"
        if not alt.exists():
            return alt
        index += 1


def parse_links(page_url: str, html: str) -> list[tuple[str, str]]:
    parser = LinkParser()
    parser.feed(html)
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for href, text in parser.links:
        absolute = normalize_url(page_url, href)
        if not absolute or absolute in seen or looks_like_asset(absolute):
            continue
        seen.add(absolute)
        results.append((absolute, text))
    # Catch bare document URLs embedded outside <a href>.
    for match in re.finditer(
        r"https?://[^\s\"'<>]+?\.(?:pdf|docx?|xlsx?|pptx?|zip|csv|txt|rtf)",
        html,
        flags=re.I,
    ):
        url = html_lib.unescape(match.group(0).rstrip(").,;]"))
        if url in seen or looks_like_asset(url):
            continue
        seen.add(url)
        results.append((url, ""))
    return results


def content_type_is_html(content_type: str) -> bool:
    return "text/html" in (content_type or "").lower()


def content_type_is_document(content_type: str, url: str) -> bool:
    ctype = (content_type or "").lower()
    if any(
        token in ctype
        for token in (
            "application/pdf",
            "application/msword",
            "application/vnd.",
            "application/zip",
            "application/octet-stream",
            "text/csv",
            "text/plain",
            "application/rtf",
        )
    ):
        # Avoid treating random binary/html-as-octet as docs unless URL looks right
        # or content-type is an explicit document type.
        if "octet-stream" in ctype:
            return looks_like_document_url(url)
        if "text/plain" in ctype:
            return looks_like_document_url(url)
        return True
    return looks_like_document_url(url)


def head_or_probe(
    opener: urllib.request.OpenerDirector, url: str
) -> tuple[str, str, int | None, str | None]:
    """Return final_url, content_type, content_length, content_disposition."""
    try:
        response, _ = request(opener, url, method="HEAD")
        with response:
            length_header = response.headers.get("Content-Length")
            length = int(length_header) if length_header and length_header.isdigit() else None
            return (
                response.geturl(),
                response.headers.get("Content-Type", ""),
                length,
                response.headers.get("Content-Disposition"),
            )
    except urllib.error.HTTPError as exc:
        # Some hosts reject HEAD; fall back to a 1-byte range GET.
        if exc.code not in {403, 405, 501}:
            raise
    response, _ = request(
        opener,
        url,
        method="GET",
        headers={"Range": "bytes=0-0"},
        read_body=True,
    )
    with response:
        length = None
        content_range = response.headers.get("Content-Range")
        if content_range and "/" in content_range:
            total = content_range.rsplit("/", 1)[-1]
            if total.isdigit():
                length = int(total)
        elif response.headers.get("Content-Length", "").isdigit():
            length = int(response.headers["Content-Length"])
        return (
            response.geturl(),
            response.headers.get("Content-Type", ""),
            length,
            response.headers.get("Content-Disposition"),
        )


def discover_documents(
    opener: urllib.request.OpenerDirector,
    start_url: str,
    *,
    delay: float,
    max_pages: int,
) -> list[Document]:
    queue: list[tuple[str, str]] = [(start_url, "Reading Room")]
    queued = {start_url}
    visited_pages: set[str] = set()
    documents: dict[str, Document] = {}

    while queue and len(visited_pages) < max_pages:
        page_url, page_label = queue.pop(0)
        if page_url in visited_pages:
            continue
        visited_pages.add(page_url)
        print(f"[page] {page_url}")

        try:
            response, _ = request(opener, page_url, read_body=False)
        except Exception as exc:  # noqa: BLE001 - keep crawling other links
            print(f"  ! failed to fetch page: {exc}", file=sys.stderr)
            continue

        with response:
            final_page_url = response.geturl()
            content_type = response.headers.get("Content-Type", "")
            content_length_header = response.headers.get("Content-Length")
            content_length = (
                int(content_length_header)
                if content_length_header and content_length_header.isdigit()
                else None
            )
            disposition = response.headers.get("Content-Disposition")

            if content_type_is_document(content_type, final_page_url):
                filename = (
                    filename_from_content_disposition(disposition)
                    or filename_from_url(final_page_url)
                )
                documents[final_page_url] = Document(
                    source_url=page_url,
                    final_url=final_page_url,
                    link_text=page_label,
                    content_type=content_type,
                    content_length=content_length,
                    filename=filename,
                )
                print(f"  -> document: {filename}")
                # Drain/close without retaining the body in memory.
                response.read(1)
                time.sleep(delay)
                continue

            if not content_type_is_html(content_type):
                print(f"  -> skipped non-html response ({content_type})")
                response.read(1)
                time.sleep(delay)
                continue

            html = response.read().decode("utf-8", errors="replace")

        for href, text in parse_links(final_page_url, html):
            if not is_same_site(href, start_url) and not looks_like_document_url(href):
                continue
            # Stay focused on the reading room and its document targets.
            if not is_same_site(href, start_url):
                continue
            if looks_like_asset(href):
                continue

            # Homepage / share / nav noise from the template.
            path = urllib.parse.urlparse(href).path.rstrip("/")
            if path in {"", "/readingroom"} and not looks_like_document_url(href):
                continue

            try:
                final_url, ctype, clen, disposition = head_or_probe(opener, href)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! probe failed for {href}: {exc}", file=sys.stderr)
                time.sleep(delay)
                continue

            if content_type_is_document(ctype, final_url) or looks_like_document_url(final_url):
                filename = (
                    filename_from_content_disposition(disposition)
                    or filename_from_url(final_url)
                )
                if final_url not in documents:
                    documents[final_url] = Document(
                        source_url=href,
                        final_url=final_url,
                        link_text=text or page_label,
                        content_type=ctype,
                        content_length=clen,
                        filename=filename,
                    )
                    print(f"  + {filename}")
            elif content_type_is_html(ctype):
                # Follow package/index pages on the same site once.
                if final_url not in visited_pages and final_url not in queued:
                    if final_url.rstrip("/") != urllib.parse.urlparse(start_url)._replace(
                        query=""
                    ).geturl().rstrip("/"):
                        # Only follow pages that look like release packages, not the whole site.
                        slug = urllib.parse.urlparse(final_url).path.strip("/").lower()
                        if slug and slug != "readingroom" and "/" not in slug:
                            queue.append((final_url, text or page_label))
                            queued.add(final_url)
            time.sleep(delay)

        time.sleep(delay)

    return sorted(documents.values(), key=lambda doc: doc.filename.lower())


def format_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown size"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def download_file(
    opener: urllib.request.OpenerDirector,
    doc: Document,
    outdir: Path,
    *,
    skip_existing: bool,
    delay: float,
) -> Path | None:
    outdir.mkdir(parents=True, exist_ok=True)
    destination = outdir / doc.filename
    if skip_existing and destination.exists() and destination.stat().st_size > 0:
        if doc.content_length is None or destination.stat().st_size == doc.content_length:
            print(f"[skip] {destination.name} (already exists)")
            return destination
        # Size mismatch: write beside the existing file.
        destination = unique_path(outdir, doc.filename)

    # Avoid collisions with unrelated existing files.
    if destination.exists():
        destination = unique_path(outdir, doc.filename)

    print(f"[get ] {doc.filename} ({format_size(doc.content_length)})")
    print(f"       {doc.final_url}")

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        response, _ = request(opener, doc.final_url, method="GET", read_body=False)
        with response:
            # Prefer server-provided name if HEAD missed it.
            cd_name = filename_from_content_disposition(
                response.headers.get("Content-Disposition")
            )
            if cd_name and cd_name != destination.name and not destination.exists():
                destination = outdir / cd_name
                tmp_path = destination.with_suffix(destination.suffix + ".part")

            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None
            bytes_read = 0
            last_report = 0
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bytes_read += len(chunk)
                    if total and bytes_read - last_report >= max(total // 20, 1):
                        pct = (bytes_read / total) * 100
                        print(f"       {pct:5.1f}% ({format_size(bytes_read)} / {format_size(total)})")
                        last_report = bytes_read
        tmp_path.replace(destination)
        print(f"[done] {destination} ({format_size(bytes_read)})")
    except Exception as exc:  # noqa: BLE001
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        print(f"[fail] {doc.filename}: {exc}", file=sys.stderr)
        return None
    finally:
        time.sleep(delay)
    return destination


def write_manifest(
    outdir: Path,
    documents: Iterable[Document],
    downloaded: list[str],
    *,
    source_url: str,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "manifest.json"
    payload = {
        "source": source_url,
        "documents": [asdict(doc) for doc in documents],
        "downloaded_files": downloaded,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all documents from the Sen. Rand Paul Reading Room."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Reading room URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Directory to store downloads (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between network requests (default: 0.5)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Safety cap on HTML pages to crawl (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover documents and print them without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    opener = build_opener()

    print(f"Discovering documents from {args.url}")
    documents = discover_documents(
        opener,
        args.url,
        delay=max(args.delay, 0.0),
        max_pages=max(args.max_pages, 1),
    )

    if not documents:
        print("No documents found.", file=sys.stderr)
        return 1

    print(f"\nFound {len(documents)} document(s):\n")
    for index, doc in enumerate(documents, start=1):
        print(f"{index:2d}. {doc.filename}")
        print(f"    link text : {doc.link_text or '(none)'}")
        print(f"    source    : {doc.source_url}")
        print(f"    final URL : {doc.final_url}")
        print(f"    size      : {format_size(doc.content_length)}")
        print(f"    type      : {doc.content_type or '(unknown)'}")
        print()

    if args.dry_run:
        print("Dry run complete; nothing downloaded.")
        return 0

    downloaded: list[str] = []
    failures = 0
    for doc in documents:
        path = download_file(
            opener,
            doc,
            args.outdir,
            skip_existing=not args.force,
            delay=max(args.delay, 0.0),
        )
        if path is None:
            failures += 1
        else:
            downloaded.append(str(path))

    manifest = write_manifest(
        args.outdir, documents, downloaded, source_url=args.url
    )
    print(f"\nManifest written to {manifest}")
    print(f"Downloaded {len(downloaded)} / {len(documents)} file(s) into {args.outdir.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
