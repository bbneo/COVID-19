"""Turn table-of-contents text into topic and article-type summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from journal_toc.dates import (
    DateHint,
    coerce_month,
    coerce_year,
    month_name,
    resolve_date_hint,
)
from journal_toc.errors import LlmError
from journal_toc.extract import (
    ExtractionError,
    chunk_text,
    extract_pages,
    has_readable_text,
    page_text,
    strip_repeated_lines,
)
from journal_toc.llm import LlmClient, parse_json_content

# Section names used by JAMA in 2019-2020, plus close labels from other journals.
TYPE_ALIASES: dict[str, str] = {
    "original investigation": "Original Investigation",
    "original investigations": "Original Investigation",
    "original research": "Original Investigation",
    "original article": "Original Investigation",
    "original articles": "Original Investigation",
    "research article": "Original Investigation",
    "article": "Original Investigation",
    "articles": "Original Investigation",
    "preliminary communication": "Preliminary Communication",
    "research letter": "Research Letter",
    "research letters": "Research Letter",
    "brief report": "Research Letter",
    "viewpoint": "Viewpoint",
    "viewpoints": "Viewpoint",
    "opinion": "Viewpoint",
    "editorial": "Editorial",
    "editorials": "Editorial",
    "review": "Review",
    "reviews": "Review",
    "review article": "Review",
    "narrative review": "Review",
    "systematic review": "Review",
    "meta-analysis": "Review",
    "meta analysis": "Review",
    "clinical review & education": "Clinical Review & Education",
    "clinical review and education": "Clinical Review & Education",
    "clinical review": "Clinical Review & Education",
    "jama clinical challenge": "JAMA Clinical Challenge",
    "clinical challenge": "JAMA Clinical Challenge",
    "diagnostic test interpretation": "Diagnostic Test Interpretation",
    "jama diagnostic test interpretation": "Diagnostic Test Interpretation",
    "jama insights": "JAMA Insights",
    "insights": "JAMA Insights",
    "jama guide to statistics and methods": "JAMA Guide to Statistics and Methods",
    "guideline": "Guideline",
    "guidelines": "Guideline",
    "clinical practice": "Guideline",
    "medical news & perspectives": "Medical News & Perspectives",
    "medical news and perspectives": "Medical News & Perspectives",
    "news & perspectives": "Medical News & Perspectives",
    "news and perspectives": "Medical News & Perspectives",
    "news": "News",
    "health agencies update": "News",
    "news from the food and drug administration": "News",
    "news from the centers for disease control and prevention": "News",
    "biotech innovations": "News",
    "the jama forum": "News",
    "comment & response": "Comment & Response",
    "comment and response": "Comment & Response",
    "comments & response": "Comment & Response",
    "comment": "Comment & Response",
    "commentary": "Comment & Response",
    "letter": "Letter",
    "letters": "Letter",
    "correspondence": "Letter",
    "patient page": "Patient Page",
    "jama patient page": "Patient Page",
    "a piece of my mind": "A Piece of My Mind",
    "piece of my mind": "A Piece of My Mind",
    "poetry": "Poetry",
    "poetry and medicine": "Poetry",
    "the arts and medicine": "Arts and Medicine",
    "arts and medicine": "Arts and Medicine",
    "correction": "Correction",
    "corrections": "Correction",
    "erratum": "Correction",
    "this week in jama": "This Week in JAMA",
    "other": "Other",
}

TYPE_ORDER: list[str] = [
    "Original Investigation",
    "Preliminary Communication",
    "Research Letter",
    "Review",
    "Guideline",
    "Clinical Review & Education",
    "JAMA Clinical Challenge",
    "Diagnostic Test Interpretation",
    "JAMA Insights",
    "JAMA Guide to Statistics and Methods",
    "Viewpoint",
    "Editorial",
    "A Piece of My Mind",
    "Medical News & Perspectives",
    "News",
    "Comment & Response",
    "Letter",
    "Patient Page",
    "Arts and Medicine",
    "Poetry",
    "This Week in JAMA",
    "Correction",
    "Other",
]

TOPIC_ALIASES: dict[str, str] = {
    "covid": "COVID-19",
    "covid-19": "COVID-19",
    "covid19": "COVID-19",
    "sars-cov-2": "COVID-19",
    "sars-cov2": "COVID-19",
    "coronavirus": "COVID-19",
    "2019 novel coronavirus": "COVID-19",
    "2019-ncov": "COVID-19",
    "ncov": "COVID-19",
}

SYSTEM_PROMPT = """You catalog articles printed on medical-journal table-of-contents pages.
The pages are often from JAMA weekly issues in 2019 and 2020. They may be another journal.
Reply with one JSON object and no other text.
Use only titles, authors, page numbers, and section headings that appear in the page text.
Do not invent study results, methods, sample sizes, or conclusions.
Skip the masthead, subscription notices, continuing-education boilerplate, and advertisements.
Skip "This Week in JAMA" blurbs that only preview an article listed again under its own section.
"""


@dataclass
class Article:
    title: str
    type: str
    topic: str
    summary: str
    authors: str | None = None
    pages: str | None = None

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "type": self.type,
            "topic": self.topic,
            "summary": self.summary,
            "authors": self.authors,
            "pages": self.pages,
        }


@dataclass
class IssueSummary:
    journal: str
    year: int
    month: int
    overview: str
    articles: list[Article] = field(default_factory=list)
    issue_date: str | None = None
    volume: str | None = None
    issue: str | None = None
    date_source: str = "model"

    @property
    def month_label(self) -> str:
        return f"{month_name(self.month)} {self.year}"

    def grouped_by_topic(self) -> list[tuple[str, list[Article]]]:
        return _group(self.articles, key=lambda article: article.topic, order=None)

    def grouped_by_type(self) -> list[tuple[str, list[Article]]]:
        return _group(self.articles, key=lambda article: article.type, order=TYPE_ORDER)

    def as_dict(self) -> dict:
        return {
            "journal": self.journal,
            "year": self.year,
            "month": self.month,
            "month_name": month_name(self.month),
            "issue_date": self.issue_date,
            "volume": self.volume,
            "issue": self.issue,
            "date_source": self.date_source,
            "overview": self.overview,
            "article_count": len(self.articles),
            "topics": [
                {
                    "topic": topic,
                    "article_count": len(items),
                    "articles": [item.as_dict() for item in items],
                }
                for topic, items in self.grouped_by_topic()
            ],
            "by_type": [
                {
                    "type": article_type,
                    "article_count": len(items),
                    "articles": [item.as_dict() for item in items],
                }
                for article_type, items in self.grouped_by_type()
            ],
        }


@dataclass
class FileSummary:
    source_file: str
    issues: list[IssueSummary]
    extracted_text: str
    last_model_response: str = ""

    def as_dict(self) -> dict:
        journal = self.issues[0].journal if self.issues else ""
        return {
            "source_file": self.source_file,
            "journal": journal,
            "issue_count": len(self.issues),
            "issues": [issue.as_dict() for issue in self.issues],
        }


def canonical_type(value: object) -> str:
    text = _clean_text(value) or "Other"
    mapped = TYPE_ALIASES.get(text.casefold())
    if mapped:
        return mapped
    if len(text) > 80:
        return "Other"
    return text


def canonical_topic(value: object) -> str:
    text = _clean_text(value) or "Unspecified"
    mapped = TOPIC_ALIASES.get(text.casefold())
    if mapped:
        return mapped
    return text


def summarize_pdf(
    path: Path,
    client: LlmClient,
    *,
    year: int | None = None,
    month: int | None = None,
    max_pages: int = 40,
    chunk_chars: int = 9000,
) -> FileSummary:
    """Extract one PDF and ask the local model to catalog its articles."""
    raw_pages = extract_pages(path, max_pages=max_pages)
    raw_text = page_text(raw_pages)
    if not has_readable_text(raw_text):
        raise ExtractionError(
            f"{path.name} has almost no extractable text. "
            "It may be a scanned image. OCR the file, then run the summarizer again."
        )
    hint = resolve_date_hint(path, raw_text, cli_year=year, cli_month=month)
    model_text = page_text(strip_repeated_lines(raw_pages))
    chunks = chunk_text(model_text, max_chars=chunk_chars)
    parsed_issues: list[IssueSummary] = []
    last_raw = ""
    for index, chunk in enumerate(chunks, start=1):
        prompt = build_user_prompt(
            path,
            chunk,
            hint,
            part=index,
            parts=len(chunks),
        )
        payload, last_raw = _complete_json(client, prompt)
        journal_hint = _journal_hint(raw_text)
        parsed_issues.extend(
            issues_from_payload(payload, journal_hint=journal_hint, hint=hint)
        )
    if not parsed_issues:
        raise LlmError(
            f"The model returned no issues for {path.name}.",
            raw=last_raw,
        )
    issues = merge_issues(parsed_issues)
    issues = apply_date_hint(issues, hint, cli_year=year, cli_month=month)
    issues = [issue for issue in issues if issue.articles]
    if not issues:
        raise LlmError(
            f"No catalogued articles in {path.name} matched the requested year and month.",
            raw=last_raw,
        )
    return FileSummary(
        source_file=str(path),
        issues=issues,
        extracted_text=raw_text,
        last_model_response=last_raw,
    )


def build_user_prompt(
    path: Path,
    text: str,
    hint: DateHint | None,
    *,
    part: int,
    parts: int,
) -> str:
    hint_line = "None. Read the issue date from the page text."
    if hint is not None:
        hint_line = (
            f"{hint.label} (source: {hint.source}). "
            "Use this when the page text agrees, and use the page text when it gives a more specific issue date."
        )
    part_line = "This is the full extracted table of contents."
    if parts > 1:
        part_line = (
            f"This is part {part} of {parts} of the same PDF. "
            "Catalog only the articles on these pages."
        )
    return f"""Source file: {path.name}
{part_line}
Issue-date hint: {hint_line}

Return JSON with this shape:
{{
  "journal": "JAMA",
  "issues": [
    {{
      "year": 2020,
      "month": 3,
      "issue_date": "2020-03-17",
      "volume": "323",
      "issue": "11",
      "overview": "Two or three sentences on what this issue's contents cover.",
      "articles": [
        {{
          "title": "Article title as printed",
          "type": "Original Investigation",
          "topic": "COVID-19",
          "summary": "One sentence on the subject, based only on the title and section.",
          "authors": "As printed, or null",
          "pages": "As printed, or null"
        }}
      ]
    }}
  ]
}}

Article type should be the journal section. Prefer these labels when they fit:
Original Investigation, Research Letter, Preliminary Communication, Viewpoint,
Editorial, Review, Clinical Review & Education, JAMA Clinical Challenge,
Diagnostic Test Interpretation, JAMA Insights, Guideline,
Medical News & Perspectives, News, Comment & Response, Letter, Patient Page,
A Piece of My Mind, Poetry, Arts and Medicine, Correction, Other.
Topic is a short clinical subject such as COVID-19, Cardiology, Oncology,
Infectious disease, Pediatrics, Surgery, Mental health, Public health, or Health policy.
Use the same topic name for articles on the same subject.
year is the issue year and month is the issue month as a number from 1 to 12.
If several issues are present, return one object per issue.

Table of contents text:
{text}
"""


def _complete_json(client: LlmClient, prompt: str) -> tuple[dict, str]:
    last_error: Exception | None = None
    last_raw = ""
    for attempt in range(2):
        user = prompt
        if attempt == 1 and last_error is not None:
            user = (
                prompt
                + "\n\nThe previous reply was not valid JSON ("
                + str(last_error)
                + "). Reply again with one JSON object only."
            )
        last_raw = client.complete(SYSTEM_PROMPT, user)
        try:
            return parse_json_content(last_raw), last_raw
        except Exception as exc:  # json.JSONDecodeError and ValueError
            last_error = exc
    raise LlmError(f"The local model did not return valid JSON. {last_error}", raw=last_raw)


def issues_from_payload(
    payload: dict,
    *,
    journal_hint: str,
    hint: DateHint | None = None,
) -> list[IssueSummary]:
    journal = _clean_text(payload.get("journal")) or journal_hint or "Unknown journal"
    raw_issues = payload.get("issues")
    if raw_issues is None and (payload.get("articles") or payload.get("year")):
        raw_issues = [payload]
    if not isinstance(raw_issues, list):
        return []
    issues: list[IssueSummary] = []
    for raw in raw_issues:
        if not isinstance(raw, dict):
            continue
        year = coerce_year(raw.get("year"))
        month = coerce_month(raw.get("month"))
        date_source = "model"
        if year is None or month is None:
            parsed = _date_from_issue_date(raw.get("issue_date"))
            if parsed is not None:
                year = year or parsed[0]
                month = month or parsed[1]
        if (year is None or month is None) and hint is not None:
            year = year or hint.year
            month = month or hint.month
            date_source = hint.source
        if year is None or month is None:
            continue
        articles = [_article_from_payload(item) for item in raw.get("articles") or []]
        articles = [article for article in articles if article is not None]
        issue_journal = _clean_text(raw.get("journal")) or journal
        issues.append(
            IssueSummary(
                journal=issue_journal,
                year=year,
                month=month,
                overview=_clean_text(raw.get("overview")) or "",
                articles=articles,
                issue_date=_normalize_issue_date(raw.get("issue_date"), year, month),
                volume=_clean_text(raw.get("volume")),
                issue=_clean_text(raw.get("issue")),
                date_source=date_source,
            )
        )
    return issues


def apply_date_hint(
    issues: list[IssueSummary],
    hint: DateHint | None,
    *,
    cli_year: int | None,
    cli_month: int | None,
) -> list[IssueSummary]:
    """Keep the model's catalog and correct the issue month when we know it."""
    if hint is None:
        return issues
    if cli_year is not None and cli_month is not None:
        if len(issues) == 1:
            _set_hint_date(issues[0], hint)
            return issues
        matched = [
            issue for issue in issues if issue.year == cli_year and issue.month == cli_month
        ]
        return matched
    if hint.confident and len(issues) == 1:
        _set_hint_date(issues[0], hint)
        return issues
    for issue in issues:
        if issue.date_source == "model":
            continue
        issue.date_source = hint.source
    return issues


def merge_issues(issues: list[IssueSummary]) -> list[IssueSummary]:
    """Combine chunk-level catalogs that describe the same issue."""
    merged: dict[tuple[str, int, int, str], IssueSummary] = {}
    order: list[tuple[str, int, int, str]] = []
    for issue in issues:
        key = (
            issue.journal.casefold(),
            issue.year,
            issue.month,
            (issue.issue or "").casefold(),
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = issue
            order.append(key)
            continue
        seen = {_title_key(article.title) for article in existing.articles}
        for article in issue.articles:
            marker = _title_key(article.title)
            if marker in seen:
                continue
            seen.add(marker)
            existing.articles.append(article)
        if issue.overview and issue.overview not in existing.overview:
            existing.overview = (existing.overview + " " + issue.overview).strip()
        existing.issue_date = existing.issue_date or issue.issue_date
        existing.volume = existing.volume or issue.volume
        existing.issue = existing.issue or issue.issue
    result = [merged[key] for key in order]
    result.sort(key=lambda issue: (issue.year, issue.month, issue.issue_date or "", issue.issue or ""))
    return result


def _set_hint_date(issue: IssueSummary, hint: DateHint) -> None:
    issue.year = hint.year
    issue.month = hint.month
    issue.date_source = hint.source
    if hint.day is not None:
        issue.issue_date = f"{hint.year:04d}-{hint.month:02d}-{hint.day:02d}"
    elif issue.issue_date and not issue.issue_date.startswith(f"{hint.year:04d}-{hint.month:02d}"):
        issue.issue_date = f"{hint.year:04d}-{hint.month:02d}"


def _article_from_payload(raw: object) -> Article | None:
    if not isinstance(raw, dict):
        return None
    title = _clean_text(raw.get("title"))
    if not title:
        return None
    summary = _clean_text(raw.get("summary")) or ""
    if summary.casefold() == title.casefold():
        summary = ""
    return Article(
        title=title,
        type=canonical_type(raw.get("type")),
        topic=canonical_topic(raw.get("topic")),
        summary=summary,
        authors=_clean_text(raw.get("authors")),
        pages=_clean_text(raw.get("pages")),
    )


def _date_from_issue_date(value: object) -> tuple[int, int, int | None] | None:
    text = _clean_text(value)
    if not text:
        return None
    iso = re.fullmatch(r"((?:19|20)\d{2})-(\d{2})(?:-(\d{2}))?", text)
    if iso:
        year = int(iso.group(1))
        month = int(iso.group(2))
        day = int(iso.group(3)) if iso.group(3) else None
        if 1 <= month <= 12 and (day is None or 1 <= day <= 31):
            return year, month, day
    from journal_toc.dates import dates_in_text

    found = dates_in_text(text, limit=80)
    if found:
        return found[0]
    return None


def _normalize_issue_date(value: object, year: int, month: int) -> str | None:
    parsed = _date_from_issue_date(value)
    if parsed is None:
        return None
    parsed_year, parsed_month, day = parsed
    use_year = parsed_year or year
    use_month = parsed_month or month
    if day:
        return f"{use_year:04d}-{use_month:02d}-{day:02d}"
    return f"{use_year:04d}-{use_month:02d}"


def _journal_hint(text: str) -> str:
    head = text[:1500].casefold()
    if "jama" in head:
        return "JAMA"
    return ""


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        return None
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n-–—")
    if not text or text.casefold() in {"null", "none", "n/a", "na"}:
        return None
    return text


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()


def _group(articles: list[Article], *, key, order: list[str] | None):
    buckets: dict[str, list[Article]] = {}
    for article in articles:
        buckets.setdefault(key(article), []).append(article)
    if order is None:
        names = sorted(buckets, key=lambda name: (-len(buckets[name]), name.casefold()))
    else:
        rank = {name: index for index, name in enumerate(order)}
        names = sorted(
            buckets,
            key=lambda name: (rank.get(name, len(order)), name.casefold()),
        )
    return [(name, buckets[name]) for name in names]


# Re-export helpers tests use for day coercion without importing dates twice.
__all__ = [
    "Article",
    "FileSummary",
    "IssueSummary",
    "apply_date_hint",
    "canonical_topic",
    "canonical_type",
    "merge_issues",
    "summarize_pdf",
]
