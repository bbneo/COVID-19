"""Tests for the medical-journal table-of-contents summarizer."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from journal_toc.cli import DEFAULT_INPUT, collect_pdfs, main
from journal_toc.dates import parse_filename_date, resolve_date_hint
from journal_toc.extract import chunk_text, strip_repeated_lines
from journal_toc.llm import parse_json_content
from journal_toc.render import render_file_markdown, render_month_index
from journal_toc.summarize import (
    Article,
    FileSummary,
    IssueSummary,
    apply_date_hint,
    canonical_topic,
    canonical_type,
    issues_from_payload,
    merge_issues,
)
from journal_toc.dates import DateHint


TOC_LINES = [
    "JAMA",
    "January 7, 2020",
    "Volume 323, Number 1",
    "Original Investigation",
    "Clinical Characteristics of Coronavirus Disease 2019 in China",
    "Guan W, Ni Z, Hu Y",
    "Pages 1-12",
    "Editorial",
    "A Novel Coronavirus Emerging in China",
    "Fauci AS",
    "Pages 13-14",
    "Research Letter",
    "Detection of SARS-CoV-2 in Different Types of Clinical Specimens",
    "Wang W",
    "Pages 15-16",
]

MODEL_CATALOG = {
    "journal": "JAMA",
    "issues": [
        {
            "year": 2020,
            "month": 1,
            "issue_date": "2020-01-07",
            "volume": "323",
            "issue": "1",
            "overview": "This issue covers early clinical reports of coronavirus disease 2019.",
            "articles": [
                {
                    "title": "Clinical Characteristics of Coronavirus Disease 2019 in China",
                    "type": "Original Investigation",
                    "topic": "COVID-19",
                    "summary": "A clinical series describing coronavirus disease 2019 in China.",
                    "authors": "Guan W, Ni Z, Hu Y",
                    "pages": "1-12",
                },
                {
                    "title": "A Novel Coronavirus Emerging in China",
                    "type": "editorial",
                    "topic": "covid-19",
                    "summary": "An editorial on a novel coronavirus emerging in China.",
                    "authors": "Fauci AS",
                    "pages": "13-14",
                },
                {
                    "title": "Detection of SARS-CoV-2 in Different Types of Clinical Specimens",
                    "type": "Research Letter",
                    "topic": "Infectious disease",
                    "summary": "A research letter on detecting SARS-CoV-2 in clinical specimens.",
                    "authors": "Wang W",
                    "pages": "15-16",
                },
            ],
        }
    ],
}


class DateTests(unittest.TestCase):
    def test_filename_iso_date(self):
        self.assertEqual(parse_filename_date(Path("JAMA-2020-03-17.pdf")), (2020, 3, 17))

    def test_filename_month_name(self):
        self.assertEqual(parse_filename_date(Path("toc-March-2020.pdf")), (2020, 3, None))

    def test_year_span_is_not_a_month(self):
        self.assertIsNone(parse_filename_date(Path("JAMA-2019-2020.pdf")))

    def test_issue_banner_in_text(self):
        hint = resolve_date_hint(Path("contents.pdf"), "\n".join(TOC_LINES))
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertEqual((hint.year, hint.month, hint.day), (2020, 1, 7))
        self.assertEqual(hint.source, "pdf_text")
        self.assertTrue(hint.confident)

    def test_filename_used_when_text_has_several_months(self):
        text = "Published online December 28, 2019\nJanuary 7, 2020\n" + ("word " * 30)
        hint = resolve_date_hint(Path("JAMA_2020-01-07_toc.pdf"), text)
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertEqual(hint.source, "filename")
        self.assertEqual((hint.year, hint.month), (2020, 1))


class CatalogTests(unittest.TestCase):
    def test_jama_type_and_topic_names(self):
        self.assertEqual(canonical_type("research letter"), "Research Letter")
        self.assertEqual(canonical_type("Comment & Response"), "Comment & Response")
        self.assertEqual(canonical_topic("sars-cov-2"), "COVID-19")

    def test_merge_dedupes_titles_and_groups(self):
        first = _issue(articles=[_article("Same Title", "Editorial", "Cardiology")])
        second = _issue(
            articles=[
                _article("Same Title", "Editorial", "Cardiology"),
                _article("Another Title", "Viewpoint", "Health policy"),
            ],
            overview="Second chunk.",
        )
        merged = merge_issues([first, second])
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            [article.title for article in merged[0].articles],
            ["Same Title", "Another Title"],
        )
        topics = [name for name, _items in merged[0].grouped_by_topic()]
        self.assertEqual(topics[0], "Cardiology")
        types = [name for name, _items in merged[0].grouped_by_type()]
        self.assertEqual(types, ["Viewpoint", "Editorial"])

    def test_cli_month_filters_a_multi_issue_file(self):
        january = _issue()
        march = _issue()
        march.month = 3
        hint = DateHint(year=2020, month=3, day=None, source="cli", confident=True)
        kept = apply_date_hint([january, march], hint, cli_year=2020, cli_month=3)
        self.assertEqual([issue.month for issue in kept], [3])

    def test_markdown_lists_topic_type_and_month(self):
        issues = issues_from_payload(MODEL_CATALOG, journal_hint="JAMA")
        summary = FileSummary(source_file="JAMA-2020-01-07.pdf", issues=issues, extracted_text="")
        markdown = render_file_markdown(summary)
        self.assertIn("# JAMA — January 2020", markdown)
        self.assertIn("### By topic", markdown)
        self.assertIn("#### COVID-19 (2)", markdown)
        self.assertIn("### By article type", markdown)
        self.assertIn("#### Original Investigation (1)", markdown)
        self.assertIn("#### Editorial (1)", markdown)
        self.assertIn("Clinical Characteristics of Coronavirus Disease 2019 in China", markdown)
        index = render_month_index([summary])
        self.assertIn("## 2020-01 January 2020", index)
        self.assertIn("COVID-19 (2)", index)

    def test_json_fence_and_flat_payload(self):
        payload = parse_json_content("```json\n{\"journal\": \"JAMA\", \"year\": 2019}\n```")
        self.assertEqual(payload["journal"], "JAMA")
        issues = issues_from_payload(
            {
                "journal": "JAMA",
                "year": "2019",
                "month": "November",
                "articles": [{"title": "Opioid Prescribing After Surgery", "type": "article"}],
            },
            journal_hint="",
        )
        self.assertEqual(issues[0].year, 2019)
        self.assertEqual(issues[0].month, 11)
        self.assertEqual(issues[0].articles[0].type, "Original Investigation")
        self.assertEqual(issues[0].articles[0].topic, "Unspecified")


class ExtractTests(unittest.TestCase):
    def test_repeated_running_header_is_kept_on_the_first_page(self):
        pages = [
            "JAMA January 7, 2020\nOriginal Investigation\nFirst title",
            "JAMA January 7, 2020\nEditorial\nSecond title",
            "JAMA January 7, 2020\nViewpoint\nThird title",
        ]
        cleaned = strip_repeated_lines(pages)
        self.assertIn("JAMA January 7, 2020", cleaned[0])
        self.assertNotIn("JAMA January 7, 2020", cleaned[1])
        self.assertIn("Second title", cleaned[1])

    def test_chunk_text_splits_on_page_breaks(self):
        text = "page one\n\f\npage two\n\f\npage three"
        chunks = chunk_text(text, max_chars=20)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("page one", chunks[0])


class CommandTests(unittest.TestCase):
    def test_default_sample_directory(self):
        self.assertEqual(
            DEFAULT_INPUT,
            Path("~/Dropbox/PublicHealth/Covid-2026/JAMA-2019-2020"),
        )

    def test_missing_sample_directory_is_an_error(self):
        with mock.patch("journal_toc.cli.DEFAULT_INPUT", Path("/tmp/missing-jama-toc-samples")):
            self.assertEqual(main([]), 2)

    def test_year_requires_month(self):
        self.assertEqual(main(["--year", "2020", "missing.pdf"]), 2)

    def test_local_model_summarizes_a_jama_toc_pdf(self):
        prompts: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                prompts.append(body["messages"][-1]["content"])
                content = json.dumps(MODEL_CATALOG)
                payload = json.dumps({"message": {"role": "assistant", "content": content}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, fmt, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pdf_path = root / "JAMA-2020-01-07.pdf"
                _write_text_pdf(pdf_path, TOC_LINES)
                output = root / "out"
                code = main(
                    [
                        str(pdf_path),
                        "--host",
                        f"http://127.0.0.1:{port}",
                        "--output",
                        str(output),
                        "--stdout",
                    ]
                )
                self.assertEqual(code, 0, prompts)
                self.assertTrue(prompts)
                self.assertIn("Clinical Characteristics of Coronavirus Disease 2019 in China", prompts[0])
                self.assertIn("January 7, 2020", prompts[0])
                markdown = (output / "JAMA-2020-01-07.summary.md").read_text(encoding="utf-8")
                self.assertIn("January 2020", markdown)
                self.assertIn("Date read from the PDF text", markdown)
                self.assertIn("COVID-19", markdown)
                self.assertIn("Original Investigation", markdown)
                self.assertIn("Research Letter", markdown)
                saved = json.loads((output / "JAMA-2020-01-07.summary.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["issues"][0]["year"], 2020)
                self.assertEqual(saved["issues"][0]["month"], 1)
                topics = [item["topic"] for item in saved["issues"][0]["topics"]]
                self.assertIn("COVID-19", topics)
                csv_text = (output / "articles.csv").read_text(encoding="utf-8")
                self.assertIn("Detection of SARS-CoV-2", csv_text)
                self.assertIn("2020-01 January 2020", (output / "by-month.md").read_text(encoding="utf-8"))
                found = collect_pdfs([str(root)], recursive=False)
                self.assertEqual(found, [pdf_path])
        finally:
            server.shutdown()
            server.server_close()


def _article(title: str, article_type: str, topic: str) -> Article:
    return Article(title=title, type=article_type, topic=topic, summary="About " + title)


def _issue(articles: list[Article] | None = None, overview: str = "Overview.") -> IssueSummary:
    return IssueSummary(
        journal="JAMA",
        year=2020,
        month=1,
        overview=overview,
        articles=articles
        or [_article("Clinical Characteristics of Coronavirus Disease 2019 in China", "Original Investigation", "COVID-19")],
        issue_date="2020-01-07",
    )


def _write_text_pdf(path: Path, lines: list[str]) -> None:
    """Write a one-page text PDF that pypdf can extract."""
    commands = ["BT", "/F1 11 Tf", "54 750 Td"]
    for index, line in enumerate(lines):
        safe = (
            line.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .encode("latin-1", "replace")
            .decode("latin-1")
        )
        if index == 0:
            commands.append(f"({safe}) Tj")
        else:
            commands.append("0 -16 Td")
            commands.append(f"({safe}) Tj")
    commands.append("ET")
    stream = "\n".join(commands).encode("latin-1")
    content = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
    pages = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    catalog = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects = [catalog, pages, page, content, font]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(output)


if __name__ == "__main__":
    unittest.main()
