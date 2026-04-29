"""
preprocessor.py
---------------
Tokenise, filter, and normalise resume text.

Uses NLTK for tokenisation, stop-word loading, and lemmatisation
exactly as taught in Week 9 of the course.

If NLTK data cannot be downloaded (e.g. restricted network), the module
falls back to a regex tokeniser and a built-in English stop-word list
so the pipeline still works without internet access.

Expects input already run through ResumeCleaner (plain ASCII, lowercase,
no special chars). Returns a single whitespace-joined token string ready
for the vectoriser.
"""

import re
import nltk
import pandas as pd

# try to download NLTK data; silently continue if blocked
for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

# NLTK tokeniser
try:
    from nltk.tokenize import word_tokenize as _nltk_word_tokenize
    _nltk_word_tokenize("test")
    _HAVE_NLTK_TOK = True
except Exception:
    _HAVE_NLTK_TOK = False

# NLTK stop words
try:
    from nltk.corpus import stopwords as _nltk_sw
    _NLTK_STOPWORDS = set(_nltk_sw.words("english"))
    _HAVE_NLTK_SW   = True
except Exception:
    _HAVE_NLTK_SW   = False
    _NLTK_STOPWORDS = set()

# NLTK lemmatizer
try:
    from nltk.stem import WordNetLemmatizer as _WNL
    _wnl = _WNL()
    _wnl.lemmatize("test")
    _HAVE_NLTK_LEM = True
except Exception:
    _HAVE_NLTK_LEM = False

# NLTK stemmer
try:
    from nltk.stem import PorterStemmer as _PS
    _ps = _PS()
    _HAVE_NLTK_STEM = True
except Exception:
    _HAVE_NLTK_STEM = False

# Built-in English stop-word fallback (used when NLTK data is unavailable)
_BUILTIN_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","then","once","here","there",
    "when","where","why","how","all","both","each","few","more","most","other",
    "some","such","no","nor","not","only","own","same","so","than","too","very",
    "can","will","just","don","should","now","ain","couldn","didn","doesn",
    "hadn","hasn","haven","isn","mightn","mustn","needn","shan","shouldn",
    "wasn","weren","won","wouldn","s","t","d","ll","m","re","ve","y",
}


def _tokenize(text):
    """Tokenise text using NLTK if available, else simple regex split."""
    if _HAVE_NLTK_TOK:
        return _nltk_word_tokenize(text)
    return re.findall(r"[a-zA-Z]+", text)


def _lemmatize(token):
    """Lemmatise a single token; fall back to identity if NLTK is unavailable."""
    if _HAVE_NLTK_LEM:
        return _wnl.lemmatize(token)
    return token


def _stem(token):
    """Stem a single token; fall back to identity if NLTK is unavailable."""
    if _HAVE_NLTK_STEM:
        return _ps.stem(token)
    return token


class ResumePreprocessor:
    """
    Convert a cleaned resume string into a normalised token string.

    Steps applied in order:
        1. word_tokenize  (NLTK punkt tokeniser, or regex fallback)
        2. Remove stop words
        3. Keep only alphabetic tokens longer than min_token_length
        4. Lemmatise (default) or stem

    Methods
    -------
    process(text)          -> str        : process a single string
    process_series(series) -> pd.Series  : process every row in a Series
    add_stopwords(words)                 : add extra words to the stop list
    """

    def __init__(
        self,
        use_lemmatization=True,
        use_stemming=False,
        min_token_length=2,
        extra_stopwords=None,
    ):
        """
        Parameters
        ----------
        use_lemmatization : bool, default True
            Map tokens to dictionary form with WordNetLemmatizer.
        use_stemming      : bool, default False
            Chop suffixes with PorterStemmer (ignored when lemmatization is on).
        min_token_length  : int, default 2
            Drop tokens shorter than this many characters.
        extra_stopwords   : list of str, optional
            Additional words to treat as stop words.
        """
        self.use_lemmatization = use_lemmatization
        self.use_stemming      = use_stemming
        self.min_token_length  = min_token_length

        self._stopwords = set(_NLTK_STOPWORDS) if _HAVE_NLTK_SW else set(_BUILTIN_STOPWORDS)
        if extra_stopwords:
            self._stopwords.update(w.lower() for w in extra_stopwords)

    def process(self, text):
        """
        Tokenise and normalise a single cleaned resume string.

        Parameters
        ----------
        text : str  — cleaned resume text (output of ResumeCleaner)

        Returns
        -------
        str  — space-joined normalised tokens (empty string if nothing left)
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        tokens = _tokenize(text)
        tokens = [t for t in tokens if t.lower() not in self._stopwords]
        tokens = [t for t in tokens if t.isalpha() and len(t) >= self.min_token_length]
        if self.use_lemmatization:
            tokens = [_lemmatize(t) for t in tokens]
        elif self.use_stemming:
            tokens = [_stem(t) for t in tokens]
        return " ".join(tokens)

    def process_series(self, series):
        """
        Apply process() to every element of a pandas Series.

        Parameters
        ----------
        series : pd.Series  — column of cleaned resume strings

        Returns
        -------
        pd.Series  — same index, processed strings
        """
        return series.apply(self.process)

    def add_stopwords(self, words):
        """
        Extend the stop-word set at runtime.

        Parameters
        ----------
        words : list of str  — words to add
        """
        self._stopwords.update(w.lower() for w in words)

    def __repr__(self):
        tok_src = "NLTK" if _HAVE_NLTK_TOK else "regex"
        sw_src  = "NLTK" if _HAVE_NLTK_SW  else "builtin"
        return (
            f"ResumePreprocessor("
            f"lemmatize={self.use_lemmatization}, "
            f"stem={self.use_stemming}, "
            f"min_len={self.min_token_length}, "
            f"tokenizer={tok_src}, stopwords={sw_src})"
        )
