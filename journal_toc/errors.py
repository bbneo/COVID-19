"""Failures the command-line tool reports directly."""


class TocError(Exception):
    """Expected failure while summarizing a table of contents."""


class LlmError(TocError):
    """The local model could not be reached or did not return JSON."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw
