"""
matcher.py
Find the most similar resumes for a given job description.

"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeMatcher:

    def __init__(self, max_features=10_000, ngram_range=(1, 2)):

        self.max_features = max_features
        self.ngram_range = ngram_range

        self._tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )

        self._X = None
        self._df = None
        self._fitted = False
        self._corpus_fingerprint = None

    def fit(self, df, text_col="processed_resume"):

        texts = df[text_col].fillna("").tolist()
        fingerprint = hash(tuple(texts))

        if self._fitted and fingerprint == self._corpus_fingerprint:
            return self

        self._df = df.reset_index(drop=True)
        self._X = self._tfidf.fit_transform(texts)
        self._fitted = True
        self._corpus_fingerprint = fingerprint

        return self

    def match(self, jd_text, top_n=5, category_filter=None):

        self._check_fitted()

        jd_vec = self._tfidf.transform([jd_text])

        sims = cosine_similarity(jd_vec, self._X).flatten()

        if category_filter and "Category" in self._df.columns:
            mask = self._df["Category"] == category_filter
            sims[~mask.values] = -1

        top_idx = sims.argsort()[::-1][:top_n]

        rows = []
        for rank, idx in enumerate(top_idx, start=1):
            row = self._df.iloc[idx]

            rows.append({
                "rank": rank,
                "Category": row.get("Category", "?"),
                "similarity_pct": round(float(sims[idx]) * 100, 1),
                "resume_snippet": str(row.get("Resume", ""))[:250] + "...",
            })

        return pd.DataFrame(rows)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before using match().")

    def __repr__(self):
        n = len(self._df) if self._fitted else "?"
        return f"ResumeMatcher(corpus_size={n}, max_features={self.max_features})"