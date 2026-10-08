"""Find an xAI API key already stored for StockAgent."""

from __future__ import annotations

import os
import re
from pathlib import Path

KEY_NAMES = ("XAI_API_KEY", "GROK_API_KEY", "LOCAL_LLM_API_KEY", "OPENAI_API_KEY")
_ASSIGNMENT = re.compile(
    r"(?im)^(?:export\s+)?(?P<name>XAI_API_KEY|GROK_API_KEY|LOCAL_LLM_API_KEY|OPENAI_API_KEY)"
    r"\s*[=:]\s*[\"']?(?P<value>[^\"'\s#]+)"
)
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    "dist",
    "build",
    ".tox",
}
_PLACEHOLDERS = {
    "changeme",
    "placeholder",
    "your-key",
    "your_key",
    "your-xai-key",
    "todo",
    "xxx",
    "none",
    "null",
}


def load_api_key() -> tuple[str, str] | None:
    """Return the key and a description of where it was found.

    Environment variables win. Otherwise the first real key in a StockAgent
    folder is used. The key text is not logged by this function.
    """
    for name in KEY_NAMES:
        value = _clean(os.environ.get(name))
        if value:
            return value, f"environment variable {name}"
    for directory in stockagent_directories():
        found = _key_in_tree(directory)
        if found is not None:
            value, name, path = found
            return value, f"{path} ({name})"
    return None


def stockagent_directories() -> list[Path]:
    """Directories named StockAgent under the usual home-folder locations."""
    home = Path.home()
    roots = [
        home,
        home / "Documents",
        home / "Desktop",
        home / "Dropbox",
        home / "wav",
        home / "projects",
        home / "Projects",
        home / "code",
        home / "src",
        home / "dev",
        home / "repos",
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        candidates = [root / "StockAgent", root / "stockagent"]
        try:
            children = list(root.iterdir())
        except OSError:
            children = []
        for child in children:
            if not child.is_dir():
                continue
            if child.name.lower() == "stockagent":
                candidates.append(child)
            elif root == home / "Dropbox":
                nested = child / "StockAgent"
                if nested.is_dir():
                    candidates.append(nested)
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(candidate)
    return found


def _key_in_tree(root: Path) -> tuple[str, str, Path] | None:
    matches: list[tuple[int, int, str, str, Path]] = []
    root_resolved = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        try:
            depth = len(current.resolve().relative_to(root_resolved).parts)
        except ValueError:
            depth = 99
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS and not name.startswith(".")]
        if depth > 3:
            dirnames[:] = []
            continue
        for filename in filenames:
            if filename.startswith(".") and filename not in {".env", ".env.local", ".envrc"}:
                continue
            path = current / filename
            if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".zip", ".pyc", ".so", ".dylib"}:
                continue
            try:
                if path.stat().st_size > 262_144:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for match in _ASSIGNMENT.finditer(text):
                value = _clean(match.group("value"))
                if not value:
                    continue
                name = match.group("name")
                if path.name in {".env", ".env.local", ".envrc"} or path.suffix == ".env":
                    env_bonus = 0
                elif "example" in path.name.casefold() or "sample" in path.name.casefold():
                    env_bonus = 2
                else:
                    env_bonus = 1
                matches.append((KEY_NAMES.index(name), env_bonus, depth, value, name, path))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    _priority, _env_bonus, _depth, value, name, path = matches[0]
    return value, name, path


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().strip('"').strip("'").strip()
    if len(text) < 8 or text.casefold() in _PLACEHOLDERS or "your-" in text.casefold():
        return None
    return text
