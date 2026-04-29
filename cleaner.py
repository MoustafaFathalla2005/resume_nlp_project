"""
cleaner.py
----------
Strip noise from raw resume text before any NLP processing.

Removes emails, phone numbers, URLs, special characters, and
normalises whitespace. Does NOT tokenise — that is preprocessor.py.
"""

import re
import unicodedata
import pandas as pd


class ResumeCleaner:
    """
    Clean raw resume text.

    Steps applied in order:
        1. Remove e-mail addresses
        2. Remove phone numbers
        3. Remove URLs
        4. Normalise unicode to plain ASCII
        5. Remove all non-alphanumeric, non-space characters
        6. Collapse multiple spaces into one
        7. Lowercase (optional)

    Methods
    -------
    clean(text)          -> str          : clean a single string
    clean_series(series) -> pd.Series    : clean every row in a Series
    """

    def __init__(self, lowercase=True):
        """
        Parameters
        ----------
        lowercase : bool, default True
            Convert text to lowercase after cleaning.
        """
        self.lowercase = lowercase

        # compile regex patterns once for speed
        self._re_email   = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")
        self._re_phone   = re.compile(r"(\+?\d[\d\s\-(). ]{7,}\d)")
        self._re_url     = re.compile(r"http[s]?://\S+|www\.\S+|\S+\.(com|org|net|io|edu|gov)\S*", re.I)
        self._re_special = re.compile(r"[^a-zA-Z0-9\s]")
        self._re_ws      = re.compile(r"\s+")

    def clean(self, text):
        """
        Clean a single resume string.

        Parameters
        ----------
        text : str  — raw resume text

        Returns
        -------
        str  — cleaned text (empty string if input is blank or not a string)
        """
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
        """
        Apply clean() to every element of a pandas Series.

        Parameters
        ----------
        series : pd.Series  — column of raw resume strings

        Returns
        -------
        pd.Series  — same index, cleaned strings
        """
        return series.apply(self.clean)

    def __repr__(self):
        return f"ResumeCleaner(lowercase={self.lowercase})"
