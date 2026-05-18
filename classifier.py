"""
classifier.py
Classify a resume into a job category using TF-IDF and cosine similarity.

This is called a Centroid Classifier — easy to understand and works
well on text data.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeClassifier:
    """
    Classify resumes by cosine similarity to per-category TF-IDF centroids.
    This replaces sklearn's LogisticRegression with a transparent,
    hand-built approach so beginners can see every step.

    Methods
    fit(df, text_col, label_col)  : learn TF-IDF weights and category centroids
    predict(text)                 : predict category for a single text
    predict_topk(text, k)         : return top-k categories with scores
    """

    def __init__(self, max_features=10_000, ngram_range=(1, 2)):
        """
        Parameters
        max_features : int, default 10 000
            Maximum vocabulary size for TF-IDF.
        ngram_range  : tuple (min_n, max_n), default (1, 2)
            Use unigrams and bigrams.
        """
        self.max_features = max_features
        self.ngram_range  = ngram_range

        self._tfidf      = TfidfVectorizer(
            max_features = max_features,
            ngram_range  = ngram_range,
            sublinear_tf = True,   
        )

        # filled after fit()
        self._category_centroids = {}   
        self._categories         = []   
        self._fitted             = False


    def fit(self, df, text_col="processed_resume", label_col="Category"):
        """
        Learn TF-IDF weights and build one centroid vector per category.

        Parameters
        df        : pd.DataFrame
        text_col  : str  — column of processed resume strings
        label_col : str  — column of category labels

        Returns : self
        """
        texts  = df[text_col].fillna("").tolist()
        labels = df[label_col].tolist()

        # step 1 – fit TF-IDF on the whole corpus
        X = self._tfidf.fit_transform(texts)   

        # step 2 – compute one centroid (average) 
        groups = defaultdict(list)
        for idx, cat in enumerate(labels):
            groups[cat].append(idx)

        for cat, indices in groups.items():
            category_matrix  = X[indices]                      
            centroid         = np.asarray(category_matrix.mean(axis=0)).flatten()
            self._category_centroids[cat] = centroid

        self._categories = sorted(self._category_centroids.keys())
        self._fitted     = True
        return self

    def predict(self, text):
        """
        Predict the best-matching category for a single processed resume.

        Parameters
        text : str  — processed resume string

        Returns : str  — predicted category name
        """
        top = self.predict_topk(text, k=1)
        return top[0]["category"]

    def predict_topk(self, text, k=5):
        """
        Return the top-k most similar categories with their scores.

        Parameters
        text : str  — processed resume string
        k    : int  — number of results to return

        Returns
        list of dict  [{"category": str, "score": float}, ...]
                      sorted by score descending, length k
        """
        self._check_fitted()

        vec = self._tfidf.transform([text])   

        scores = {}
        for cat, centroid in self._category_centroids.items():
            sim = cosine_similarity(vec, centroid.reshape(1, -1))[0][0]
            scores[cat] = float(sim)

        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"category": cat, "score": score} for cat, score in sorted_cats[:k]]

    def get_categories(self):
        """
        Return the sorted list of all known categories.

        Returns
        list of str
        """
        self._check_fitted()
        return self._categories


    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before using predict().")

    def __repr__(self):
        n = len(self._categories) if self._fitted else "?"
        return (
            f"ResumeClassifier(max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}, categories={n})"
        )
