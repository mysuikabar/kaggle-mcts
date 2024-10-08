import re
import string

import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from typing_extensions import Self

from .base import BaseFittableProcessor
from .consts import STOP_WORDS


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


class TfidfProcessor(BaseFittableProcessor):
    """
    tf-idf processor for text columns
    """

    def __init__(self, max_features: int) -> None:
        self._processor = TfidfVectorizer(max_features=max_features)

    def fit(self, sr: pd.Series) -> Self:
        sr = sr.apply(_preprocess_text)
        self._processor.fit(sr)
        return self

    def transform(self, sr: pd.Series) -> pd.DataFrame:
        sr = sr.apply(_preprocess_text)
        features = self._processor.transform(sr).toarray()
        columns = self._processor.get_feature_names_out().tolist()
        return pd.DataFrame(features, index=sr.index, columns=columns)

    def fit_transform(self, sr: pd.Series) -> pd.DataFrame:
        return self.fit(sr).transform(sr)


def _fit_tfidf(sr: pd.Series, max_features: int) -> TfidfProcessor:
    processor = TfidfProcessor(max_features)
    return processor.fit(sr)


def parallel_fit_tfidf(
    df: pd.DataFrame, text_cols: list[str], max_features: int
) -> dict[str, TfidfProcessor]:
    results = Parallel(n_jobs=-1)(
        delayed(_fit_tfidf)(df[feature], max_features) for feature in text_cols
    )
    return dict(zip(text_cols, results))


def parallel_transform_tfidf(
    df: pd.DataFrame, processors: dict[str, TfidfProcessor]
) -> pd.DataFrame:
    results = Parallel(n_jobs=-1)(
        delayed(processor.transform)(df[feature])
        for feature, processor in processors.items()
    )

    # カラム名を被らないように feature_col　にしてconcat
    for feature, result in zip(processors.keys(), results):
        result.columns = [f"{feature}_{col}" for col in result.columns]

    return pd.concat(results, axis=1)
