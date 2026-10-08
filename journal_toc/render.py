"""Write per-file summaries and a month index."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from journal_toc.summarize import FileSummary, IssueSummary


def render_file_markdown(summary: FileSummary) -> str:
    source = Path(summary.source_file).name
    lines: list[str] = []
    if len(summary.issues) == 1:
        issue = summary.issues[0]
        lines.append(f"# {issue.journal} — {issue.month_label}")
        lines.append("")
        lines.extend(_issue_header_lines(issue, source))
        lines.extend(_issue_body(issue))
    else:
        journal = summary.issues[0].journal if summary.issues else "Journal"
        lines.append(f"# {journal} table of contents")
        lines.append("")
        lines.append(f"Source file: `{source}`")
        lines.append("")
        lines.append(f"This file contains {len(summary.issues)} issues.")
        lines.append("")
        for issue in summary.issues:
            lines.append(f"## {issue.month_label}")
            lines.append("")
            lines.extend(_issue_header_lines(issue, source, include_source=False))
            lines.extend(_issue_body(issue))
    return "\n".join(lines).rstrip() + "\n"


def render_month_index(summaries: list[FileSummary]) -> str:
    lines = ["# Journal contents by year and month", ""]
    buckets: dict[tuple[int, int], list[tuple[FileSummary, IssueSummary]]] = {}
    for summary in summaries:
        for issue in summary.issues:
            buckets.setdefault((issue.year, issue.month), []).append((summary, issue))
    for year, month in sorted(buckets):
        items = buckets[(year, month)]
        label = items[0][1].month_label
        lines.append(f"## {year:04d}-{month:02d} {label}")
        lines.append("")
        topic_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for _summary, issue in items:
            for article in issue.articles:
                topic_counts[article.topic] = topic_counts.get(article.topic, 0) + 1
                type_counts[article.type] = type_counts.get(article.type, 0) + 1
        lines.append(f"Issues: {len(items)}. Articles: {sum(type_counts.values())}.")
        lines.append("")
        lines.append("Topics: " + _count_phrase(topic_counts))
        lines.append("")
        lines.append("Article types: " + _count_phrase(type_counts))
        lines.append("")
        for summary, issue in items:
            name = Path(summary.source_file).name
            lines.append(
                f"- `{name}` — {issue.journal}, {len(issue.articles)} articles"
                + (f", {issue.issue_date}" if issue.issue_date else "")
            )
        lines.append("")
    if not buckets:
        lines.append("No issues were summarized.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def article_rows(summaries: list[FileSummary]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in summaries:
        source = str(Path(summary.source_file))
        for issue in summary.issues:
            for article in issue.articles:
                rows.append(
                    {
                        "source_file": source,
                        "journal": issue.journal,
                        "year": str(issue.year),
                        "month": f"{issue.month:02d}",
                        "month_name": issue.month_label.split()[0],
                        "issue_date": issue.issue_date or "",
                        "volume": issue.volume or "",
                        "issue": issue.issue or "",
                        "type": article.type,
                        "topic": article.topic,
                        "title": article.title,
                        "authors": article.authors or "",
                        "pages": article.pages or "",
                        "summary": article.summary,
                    }
                )
    return rows


def write_summaries(
    summaries: list[FileSummary],
    output_dir: Path,
    *,
    save_text: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    used: set[str] = set()
    for summary in summaries:
        stem = _unique_stem(Path(summary.source_file).stem, used)
        json_path = output_dir / f"{stem}.summary.json"
        md_path = output_dir / f"{stem}.summary.md"
        json_path.write_text(
            json.dumps(summary.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(render_file_markdown(summary), encoding="utf-8")
        written.extend([json_path, md_path])
        if save_text:
            text_path = output_dir / f"{stem}.extracted.txt"
            text_path.write_text(summary.extracted_text, encoding="utf-8")
            written.append(text_path)
    index_path = output_dir / "by-month.md"
    index_path.write_text(render_month_index(summaries), encoding="utf-8")
    written.append(index_path)
    csv_path = output_dir / "articles.csv"
    rows = article_rows(summaries)
    fieldnames = [
        "source_file",
        "journal",
        "year",
        "month",
        "month_name",
        "issue_date",
        "volume",
        "issue",
        "type",
        "topic",
        "title",
        "authors",
        "pages",
        "summary",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    written.append(csv_path)
    return written


def write_model_error(output_dir: Path, source: Path, raw: str) -> Path | None:
    if not raw:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_safe_stem(source.stem)}.llm-response.txt"
    path.write_text(raw, encoding="utf-8")
    return path


def _issue_header_lines(
    issue: IssueSummary,
    source: str,
    *,
    include_source: bool = True,
) -> list[str]:
    lines: list[str] = []
    if include_source:
        lines.append(f"Source file: `{source}`")
    details: list[str] = []
    if issue.issue_date:
        details.append(f"Issue date: {issue.issue_date}")
    if issue.volume:
        details.append(f"Volume {issue.volume}")
    if issue.issue:
        details.append(f"Issue {issue.issue}")
    details.append(f"Date read from {_date_source_label(issue.date_source)}")
    lines.append(" · ".join(details))
    lines.append("")
    if issue.overview:
        lines.append(issue.overview)
        lines.append("")
    lines.append(f"Articles: {len(issue.articles)}")
    lines.append("")
    return lines


def _issue_body(issue: IssueSummary) -> list[str]:
    lines = ["### By topic", ""]
    for topic, articles in issue.grouped_by_topic():
        lines.append(f"#### {topic} ({len(articles)})")
        lines.append("")
        for article in articles:
            lines.append(_article_bullet(article, lead=article.type))
        lines.append("")
    lines.append("### By article type")
    lines.append("")
    for article_type, articles in issue.grouped_by_type():
        lines.append(f"#### {article_type} ({len(articles)})")
        lines.append("")
        for article in articles:
            lines.append(_article_bullet(article, lead=article.topic))
        lines.append("")
    return lines


def _article_bullet(article, *, lead: str) -> str:
    sentence = f"**{lead}** — {article.title}"
    if article.summary:
        sentence += f". {article.summary}"
    extras: list[str] = []
    if article.authors:
        extras.append(article.authors)
    if article.pages:
        extras.append(f"p. {article.pages}")
    if extras:
        sentence += " (" + "; ".join(extras) + ")"
    return f"- {sentence}"


def _date_source_label(source: str) -> str:
    labels = {
        "pdf_text": "the PDF text",
        "filename": "the file name",
        "cli": "--year and --month",
        "model": "the model",
    }
    return labels.get(source, source.replace("_", " "))


def _count_phrase(counts: dict[str, int]) -> str:
    ordered = sorted(counts, key=lambda name: (-counts[name], name.casefold()))
    return ", ".join(f"{name} ({counts[name]})" for name in ordered)


def _safe_stem(stem: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ".-" else "-" for char in stem)
    cleaned = cleaned.strip("-.") or "toc"
    return cleaned[:80]


def _unique_stem(stem: str, used: set[str]) -> str:
    base = _safe_stem(stem)
    candidate = base
    number = 2
    while candidate.casefold() in used:
        suffix = f"-{number}"
        candidate = base[: 80 - len(suffix)] + suffix
        number += 1
    used.add(candidate.casefold())
    return candidate
