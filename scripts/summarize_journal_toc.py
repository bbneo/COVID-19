#!/usr/bin/env python3
"""Summarize medical-journal table-of-contents PDFs with a local LLM.

Sample JAMA issues (2019-2020) live in::

    ~/Dropbox/PublicHealth/Covid-2026/JAMA-2019-2020

Examples::

    python scripts/summarize_journal_toc.py \\
        ~/Dropbox/PublicHealth/Covid-2026/JAMA-2019-2020

    python scripts/summarize_journal_toc.py \\
        ~/Dropbox/PublicHealth/Covid-2026/JAMA-2019-2020/JAMA-2020-03-17.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from journal_toc.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
