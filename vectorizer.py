import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


class BagOfNGrams:

    def __init__(
        self,
        ngram_range=(1, 2),
        max_features=20_000,
        min_df=2,
        max_df=0.95,
    ):

        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df

        self._vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
        )

        self._label_enc = LabelEncoder()

        self._X = None
        self._y = None
        self._fitted = False

    def fit(self, df, text_col="processed_resume", label_col="Category"):

        for col in (text_col, label_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        texts = df[text_col].fillna("").tolist()
        labels = df[label_col].tolist()

        self._X = self._vectorizer.fit_transform(texts)
        self._y = self._label_enc.fit_transform(labels)

        self._fitted = True

        return self

    def transform(self, texts):

        self._check_fitted()

        return self._vectorizer.transform(texts)

    def get_features(self):

        self._check_fitted()

        return self._X, self._y

    def get_feature_names(self):

        self._check_fitted()

        return self._vectorizer.get_feature_names_out().tolist()

    def get_label_mapping(self):

        self._check_fitted()

        return dict(enumerate(self._label_enc.classes_))

    def get_top_ngrams(self, category=None, top_n=20):

        self._check_fitted()

        feature_names = np.array(self.get_feature_names())

        if category is not None:
            cat_int = self._label_enc.transform([category])[0]
            mask = self._y == cat_int
            counts = np.asarray(self._X[mask].sum(axis=0)).flatten()

        else:
            counts = np.asarray(self._X.sum(axis=0)).flatten()

        top_idx = counts.argsort()[::-1][:top_n]

        return pd.DataFrame({
            "ngram": feature_names[top_idx],
            "count": counts[top_idx],
        })

    def summary(self):

        self._check_fitted()

        print("BagOfNGrams summary")
        print(f"  n-gram range  : {self.ngram_range}")
        print(f"  vocabulary    : {len(self.get_feature_names()):,} features")
        print(f"  documents     : {self._X.shape[0]:,}")
        print(f"  categories    : {list(self._label_enc.classes_)}")
        print(f"  matrix shape  : {self._X.shape}")

    def _check_fitted(self):

        if not self._fitted:
            raise RuntimeError("Call fit() before using this method.")

    def __repr__(self):

        return (
            f"BagOfNGrams(ngram_range={self.ngram_range}, "
            f"max_features={self.max_features})"
        )