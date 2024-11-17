import re
import string
from logging import getLogger

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from typing_extensions import Self

from .consts import STOP_WORDS

logger = getLogger(__name__)


class IdentityTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y: None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class CategoricalConverter(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
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

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        return self.columns


class ColumnDropper(TransformerMixin, BaseEstimator):
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns, errors="ignore")

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        if input_features is None:
            raise ValueError("input_features must be provided")
        return [col for col in input_features if col not in self.columns]


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

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        return self._vectorizer.get_feature_names_out()


class TabularDataTransformer(TransformerMixin, BaseEstimator):
    """
    - Standardize numerical features with QuantileTransformer
    - Encode categorical features with OrdinalEncoder
    """

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: None = None) -> Self:
        self.numerical_columns_ = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_columns_ = X.select_dtypes(exclude=np.number).columns.tolist()
        assert len(self.numerical_columns_) + len(self.categorical_columns_) == X.shape[1], "All columns must be either numerical or categorical"

        self._preprocessor = ColumnTransformer(
            [
                ("numerical", QuantileTransformer(output_distribution="normal", random_state=self.random_state), self.numerical_columns_),
                ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), self.categorical_columns_),
            ]
        ).set_output(transform="pandas")
        self._preprocessor.fit(X)

        # save number of categories for each categorical
        cat_encoder = self._preprocessor.named_transformers_["categorical"]
        self.n_categories_ = {col: len(cats) + 1 for col, cats in zip(self.categorical_columns_, cat_encoder.categories_)}  # +1 for unknown value

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self._preprocessor.transform(X)

        # add 1 to categorical values to avoid minus value (for nn.Embedding)
        n_cat_cols = len(self.categorical_columns_)
        if n_cat_cols > 0:
            transformed.iloc[:, -n_cat_cols:] += 1

        return pd.DataFrame(transformed.values, index=X.index, columns=self.numerical_columns_ + self.categorical_columns_)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        return self.numerical_columns_ + self.categorical_columns_
