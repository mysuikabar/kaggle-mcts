import re
import string
from logging import getLogger

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from typing_extensions import Self

from .consts import STOP_WORDS

logger = getLogger(__name__)


def _preprocess_text(text: str) -> str:
    """
    preprocess text
    """
    text = str(text).lower()

    # replace punctuation with space
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text = text.translate(translator)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords
    text = " ".join([word for word in text.split() if word not in STOP_WORDS])

    return text


class Tfidf(TransformerMixin, BaseEstimator):
    """
    tf-idf processor for text columns
    """

    def __init__(self, max_features: int) -> None:
        self._vectorizer = TfidfVectorizer(max_features=max_features)

    def fit(self, X: pd.Series, y: None = None) -> Self:
        X = X.apply(_preprocess_text)

        logger.info(f"Fitting tf-idf vectorizer for {X.name}")
        self._vectorizer.fit(X)
        logger.info(f"Fitting tf-idf vectorizer done for {X.name}")

        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        X = X.apply(_preprocess_text)

        logger.info(f"Transforming text to tf-idf features for {X.name}")
        features = self._vectorizer.transform(X).toarray()
        logger.info(f"Transforming text to tf-idf features done for {X.name}")

        columns = self._vectorizer.get_feature_names_out().tolist()
        columns = [f"tfidf_{X.name}_{col}" for col in columns]

        return pd.DataFrame(features, index=X.index, columns=columns)
