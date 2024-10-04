import re
import string

import pandas as pd
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

    def __init__(self) -> None:
        self._processor = TfidfVectorizer()

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
