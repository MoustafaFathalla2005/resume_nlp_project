"""
matcher.py
----------
Find the most similar resumes for a given job description.

Uses TF-IDF vectors and cosine similarity — the same technique from
the classifier but applied in reverse: we have a JD and want to rank
all resumes by how well they match.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeMatcher:
    """
    Rank resumes by cosine similarity to a job description.

    The matcher fits TF-IDF on the whole resume corpus, then at query
    time transforms the job description with the same vocabulary and
    computes similarity to every resume.

    Methods
    -------
    fit(df, text_col)          : build TF-IDF index on the corpus
    match(jd_text, top_n, cat) : return top-N matching resumes
    """

    def __init__(self, max_features=10_000, ngram_range=(1, 2)):
        """
        Parameters
        ----------
        max_features : int, default 10 000
        ngram_range  : tuple (min_n, max_n), default (1, 2)
        """
        self.max_features = max_features
        self.ngram_range  = ngram_range

        self._tfidf   = TfidfVectorizer(
            max_features = max_features,
            ngram_range  = ngram_range,
            sublinear_tf = True,
        )
        self._X      = None   # corpus TF-IDF matrix (n_docs, n_features)
        self._df     = None   # reference to the original DataFrame
        self._fitted = False

    # ── public ──────────────────────────────────────────────────────────────

    def fit(self, df, text_col="processed_resume"):
        """
        Build the TF-IDF index on the resume corpus.

        Parameters
        ----------
        df       : pd.DataFrame  — must contain text_col
        text_col : str           — column of processed resume strings

        Returns
        -------
        self
        """
        self._df  = df.reset_index(drop=True)
        texts     = self._df[text_col].fillna("").tolist()
        self._X   = self._tfidf.fit_transform(texts)
        self._fitted = True
        return self

    def match(self, jd_text, top_n=5, category_filter=None):
        """
        Find the top-N resumes most similar to a job description.

        Parameters
        ----------
        jd_text         : str       — job description (processed or raw)
        top_n           : int       — how many results to return
        category_filter : str|None  — restrict results to this category

        Returns
        -------
        pd.DataFrame  columns ['rank', 'Category', 'similarity_pct', 'resume_snippet']
                      sorted by similarity descending
        """
        self._check_fitted()

        # transform the JD with the corpus vocabulary
        jd_vec = self._tfidf.transform([jd_text])

        # cosine similarity against every resume
        sims = cosine_similarity(jd_vec, self._X).flatten()

        # optionally zero-out resumes that don't match the category filter
        if category_filter and "Category" in self._df.columns:
            mask = self._df["Category"] == category_filter
            sims[~mask.values] = -1

        # get top-N indices sorted by similarity descending
        top_idx = sims.argsort()[::-1][:top_n]

        rows = []
        for rank, idx in enumerate(top_idx, start=1):
            row = self._df.iloc[idx]
            rows.append({
                "rank"           : rank,
                "Category"       : row.get("Category", "?"),
                "similarity_pct" : round(float(sims[idx]) * 100, 1),
                "resume_snippet" : str(row.get("Resume", ""))[:250] + "...",
            })

        return pd.DataFrame(rows)

    # ── private ─────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before using match().")

    def __repr__(self):
        n = len(self._df) if self._fitted else "?"
        return f"ResumeMatcher(corpus_size={n}, max_features={self.max_features})"
