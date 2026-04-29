"""
vectorizer.py
-------------
Build a Bag-of-N-Grams matrix from processed resume text.

Uses CountVectorizer (sklearn) exactly as discussed in the course.
Everything is done manually: fit, transform, top-ngrams, label encoding.

No magic — just counting word frequencies.
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


class BagOfNGrams:
    """
    Fit a Bag-of-N-Grams (BoW) model on a resume DataFrame.

    Internally wraps CountVectorizer but exposes simple helper methods
    so beginners can explore the vocabulary and top n-grams easily.

    Methods
    -------
    fit(df, text_col, label_col)   : learn vocabulary and encode labels
    transform(texts)               : convert new texts to feature matrix
    get_features()                 : return (X sparse matrix, y int array)
    get_feature_names()            : list of n-gram strings in vocab order
    get_label_mapping()            : {int: category_name} dict
    get_top_ngrams(category, top_n): most frequent n-grams (globally or per category)
    summary()                      : print a quick overview
    """

    def __init__(
        self,
        ngram_range=(1, 2),
        max_features=20_000,
        min_df=2,
        max_df=0.95,
    ):
        """
        Parameters
        ----------
        ngram_range  : tuple (min_n, max_n), default (1,2)
            Include unigrams and bigrams.
        max_features : int, default 20 000
            Only keep the top-N most frequent n-grams.
        min_df       : int, default 2
            Ignore n-grams that appear in fewer than this many documents.
        max_df       : float, default 0.95
            Ignore n-grams that appear in more than 95 % of documents.
        """
        self.ngram_range  = ngram_range
        self.max_features = max_features
        self.min_df       = min_df
        self.max_df       = max_df

        self._vectorizer = CountVectorizer(
            ngram_range  = ngram_range,
            max_features = max_features,
            min_df       = min_df,
            max_df       = max_df,
        )
        self._label_enc = LabelEncoder()

        # set after fit()
        self._X       = None
        self._y       = None
        self._fitted  = False

    # ── public ──────────────────────────────────────────────────────────────

    def fit(self, df, text_col="processed_resume", label_col="Category"):
        """
        Learn vocabulary from the corpus and encode category labels.

        Parameters
        ----------
        df        : pd.DataFrame  — must contain text_col and label_col
        text_col  : str           — column of processed resume strings
        label_col : str           — column of category names

        Returns
        -------
        self  (so you can chain: bg.fit(df).summary())
        """
        for col in (text_col, label_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        texts  = df[text_col].fillna("").tolist()
        labels = df[label_col].tolist()

        self._X      = self._vectorizer.fit_transform(texts)
        self._y      = self._label_enc.fit_transform(labels)
        self._fitted = True
        return self

    def transform(self, texts):
        """
        Convert new texts to the learned feature space.

        Parameters
        ----------
        texts : list of str  — processed resume strings

        Returns
        -------
        sparse matrix  shape (len(texts), n_features)
        """
        self._check_fitted()
        return self._vectorizer.transform(texts)

    def get_features(self):
        """
        Return the fitted feature matrix and integer labels.

        Returns
        -------
        X : sparse matrix  shape (n_docs, n_features)
        y : np.ndarray     shape (n_docs,)  integer-encoded categories
        """
        self._check_fitted()
        return self._X, self._y

    def get_feature_names(self):
        """
        Return the list of n-gram strings in vocabulary order.

        Returns
        -------
        list of str
        """
        self._check_fitted()
        return self._vectorizer.get_feature_names_out().tolist()

    def get_label_mapping(self):
        """
        Return a dict mapping integer label → category name.

        Returns
        -------
        dict  {int: str}
        """
        self._check_fitted()
        return dict(enumerate(self._label_enc.classes_))

    def get_top_ngrams(self, category=None, top_n=20):
        """
        Return the most frequent n-grams, globally or per category.

        Parameters
        ----------
        category : str or None
            If None, shows global top n-grams across all resumes.
        top_n    : int, default 20
            How many n-grams to return.

        Returns
        -------
        pd.DataFrame  columns ['ngram', 'count']  sorted descending
        """
        self._check_fitted()
        feature_names = np.array(self.get_feature_names())

        if category is not None:
            cat_int = self._label_enc.transform([category])[0]
            mask    = self._y == cat_int
            counts  = np.asarray(self._X[mask].sum(axis=0)).flatten()
        else:
            counts = np.asarray(self._X.sum(axis=0)).flatten()

        top_idx = counts.argsort()[::-1][:top_n]
        return pd.DataFrame({
            "ngram": feature_names[top_idx],
            "count": counts[top_idx],
        })

    def summary(self):
        """Print a quick overview of the fitted vectoriser."""
        self._check_fitted()
        print("BagOfNGrams summary")
        print(f"  n-gram range  : {self.ngram_range}")
        print(f"  vocabulary    : {len(self.get_feature_names()):,} features")
        print(f"  documents     : {self._X.shape[0]:,}")
        print(f"  categories    : {list(self._label_enc.classes_)}")
        print(f"  matrix shape  : {self._X.shape}")

    # ── private ─────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before using this method.")

    def __repr__(self):
        return (
            f"BagOfNGrams(ngram_range={self.ngram_range}, "
            f"max_features={self.max_features})"
        )
