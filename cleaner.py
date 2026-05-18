"""
cleaner.py
Strip noise from raw resume text before any NLP processing.

Removes emails, phone numbers, URLs, special characters, and
normalises whitespace. Does NOT tokenise — that is preprocessor.py.
"""

import re
import unicodedata
import pandas as pd


class ResumeCleaner:

    def __init__(self, lowercase=True):

        self.lowercase = lowercase
        self._re_email   = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")
        self._re_phone   = re.compile(r"(\+?\d[\d\s\-(). ]{7,}\d)")
        self._re_url     = re.compile(r"http[s]?://\S+|www\.\S+|\S+\.(com|org|net|io|edu|gov)\S*", re.I)
        self._re_special = re.compile(r"[^a-zA-Z0-9\s]")
        self._re_ws      = re.compile(r"\s+")

    def clean(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self._re_email.sub(" ", text)
        text = self._re_phone.sub(" ", text)
        text = self._re_url.sub(" ", text)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = self._re_special.sub(" ", text)
        text = self._re_ws.sub(" ", text)

        if self.lowercase:
            text = text.lower()

        return text.strip()

    def clean_series(self, series):
        return series.apply(self.clean)

    def __repr__(self):
        return f"ResumeCleaner(lowercase={self.lowercase})"
