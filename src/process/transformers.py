import re
import string
from logging import getLogger

import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from typing_extensions import Self

from .consts import STOP_WORDS

logger = getLogger(__name__)


class IdentityTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y: None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class CategoricalConverter(OneToOneFeatureMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y: None = None) -> Self:
        self._cat_mapping = {
            feature: "category"
            for feature in X.columns[X.dtypes == object]  # noqa: E721
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.astype(self._cat_mapping)


class ColumnSelector(TransformerMixin, BaseEstimator):
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.filter(self.columns)


class ColumnDropper(TransformerMixin, BaseEstimator):
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns, errors="ignore")


class Tfidf(TransformerMixin, BaseEstimator):
    def __init__(self, max_features: int) -> None:
        self.max_features = max_features
        self._vectorizer = TfidfVectorizer(max_features=max_features)

    @staticmethod
    def _preprocess_text(text: str) -> str:
        text = str(text).lower()

        # replace punctuation with space
        translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
        text = text.translate(translator)

        # remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        # remove stopwords
        text = " ".join([word for word in text.split() if word not in STOP_WORDS])

        return text

    def fit(self, X: pd.Series, y: None = None) -> Self:
        X = X.apply(self._preprocess_text)

        logger.info(f"Fitting tf-idf vectorizer for {X.name}")
        self._vectorizer.fit(X)
        logger.info(f"Fitting tf-idf vectorizer done for {X.name}")

        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        X = X.apply(self._preprocess_text)

        logger.info(f"Transforming text to tf-idf features for {X.name}")
        features = self._vectorizer.transform(X).toarray()
        logger.info(f"Transforming text to tf-idf features done for {X.name}")

        columns = self._vectorizer.get_feature_names_out().tolist()

        return pd.DataFrame(features, index=X.index, columns=columns)
