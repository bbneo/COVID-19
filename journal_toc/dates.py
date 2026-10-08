"""Find the issue year and month for a table-of-contents file."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from pathlib import Path

MONTH_NAMES: dict[str, int] = {}
for _number, _name in enumerate(calendar.month_name):
    if _name:
        MONTH_NAMES[_name.lower()] = _number
for _number, _name in enumerate(calendar.month_abbr):
    if _name:
        MONTH_NAMES[_name.lower()] = _number

_MONTH_PATTERN = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
)

# Require the match to end before another digit so "2019-2020" is a span,
# not March of 2019 or February of 2020.
_FILENAME_ISO = re.compile(
    r"(?P<year>(?:19|20)\d{2})[-_.](?P<month>0?[1-9]|1[0-2])"
    r"(?:[-_.](?P<day>0?[1-9]|[12]\d|3[01]))?(?!\d)"
)
_FILENAME_MONTH = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\.?"
    rf"(?:[-_.\s]+(?P<day>0?[1-9]|[12]\d|3[01])(?!\d))?"
    rf"[-_.\s]+(?P<year>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_TEXT_MDY = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\.?\s+"
    rf"(?:(?P<day>[0-3]?\d)(?:st|nd|rd|th)?,?\s+)?"
    rf"(?P<year>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_TEXT_DMY = re.compile(
    rf"\b(?P<day>[0-3]?\d)\s+(?P<month>{_MONTH_PATTERN})\.?\s+"
    rf"(?P<year>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DateHint:
    """Year and month chosen before the model runs."""

    year: int
    month: int
    day: int | None
    source: str
    confident: bool

    @property
    def label(self) -> str:
        month_name = calendar.month_name[self.month]
        if self.day:
            return f"{month_name} {self.day}, {self.year}"
        return f"{month_name} {self.year}"


def coerce_month(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 12 else None
    if isinstance(value, float) and value.is_integer():
        number = int(value)
        return number if 1 <= number <= 12 else None
    if isinstance(value, str):
        text = value.strip().rstrip(".")
        if not text:
            return None
        if text.isdigit():
            number = int(text)
            return number if 1 <= number <= 12 else None
        return MONTH_NAMES.get(text.lower())
    return None


def coerce_year(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 1900 <= value <= 2100:
        return value
    if isinstance(value, float) and value.is_integer():
        number = int(value)
        return number if 1900 <= number <= 2100 else None
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"(?:19|20)\d{2}", text):
            return int(text)
    return None


def coerce_day(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    number: int | None = None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    if number is not None and 1 <= number <= 31:
        return number
    return None


def month_name(month: int) -> str:
    if 1 <= month <= 12:
        return calendar.month_name[month]
    return str(month)


def parse_filename_date(path: Path) -> tuple[int, int, int | None] | None:
    """Return year, month, day from a file name, when the name contains one."""
    stem = path.stem.replace("_", " ")
    iso_matches = list(_FILENAME_ISO.finditer(stem))
    named_matches = list(_FILENAME_MONTH.finditer(stem))
    candidates: list[tuple[int, int, int | None, int]] = []
    for match in iso_matches:
        year = int(match.group("year"))
        month = int(match.group("month"))
        day = int(match.group("day")) if match.group("day") else None
        if 1 <= month <= 12:
            candidates.append((year, month, day, match.start()))
    for match in named_matches:
        month = MONTH_NAMES.get(match.group("month").lower().rstrip("."))
        year = int(match.group("year"))
        day = int(match.group("day")) if match.group("day") else None
        if month:
            candidates.append((year, month, day, match.start()))
    if not candidates:
        return None
    # A name like "JAMA-2020-03-17" should win over a bare year elsewhere.
    with_day = [item for item in candidates if item[2] is not None]
    pool = with_day or candidates
    year, month, day, _start = pool[-1]
    return year, month, day


def dates_in_text(text: str, *, limit: int = 4000) -> list[tuple[int, int, int | None]]:
    """Return issue-style dates from the start of extracted TOC text."""
    head = text[:limit]
    found: list[tuple[int, int, int | None, int]] = []
    for pattern in (_TEXT_MDY, _TEXT_DMY):
        for match in pattern.finditer(head):
            month = MONTH_NAMES.get(match.group("month").lower().rstrip("."))
            year = int(match.group("year"))
            day_text = match.group("day")
            day = int(day_text) if day_text else None
            if month and (day is None or 1 <= day <= 31):
                found.append((year, month, day, match.start()))
    found.sort(key=lambda item: item[3])
    return [(year, month, day) for year, month, day, _start in found]


def hint_from_text(text: str) -> DateHint | None:
    found = dates_in_text(text)
    if not found:
        return None
    # Full dates (with a day) are issue banners. Month-year mentions are weaker.
    full = [item for item in found if item[2] is not None]
    pool = full or found
    first_year, first_month, first_day = pool[0]
    same_month = [item for item in pool if item[0] == first_year and item[1] == first_month]
    confident = bool(full) and len(same_month) == len(pool)
    day = first_day if confident else None
    return DateHint(
        year=first_year,
        month=first_month,
        day=day,
        source="pdf_text",
        confident=confident,
    )


def resolve_date_hint(
    path: Path,
    text: str,
    *,
    cli_year: int | None = None,
    cli_month: int | None = None,
) -> DateHint | None:
    """Pick the issue date to show the model and to check its answer."""
    if cli_year is not None and cli_month is not None:
        return DateHint(
            year=cli_year,
            month=cli_month,
            day=None,
            source="cli",
            confident=True,
        )
    text_hint = hint_from_text(text)
    if text_hint is not None and text_hint.confident:
        return text_hint
    filename = parse_filename_date(path)
    if filename is not None:
        year, month, day = filename
        return DateHint(
            year=year,
            month=month,
            day=day,
            source="filename",
            confident=True,
        )
    return text_hint
