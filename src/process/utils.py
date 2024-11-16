import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from typing_extensions import Self


class CategoricalConverter(OneToOneFeatureMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y: None = None) -> Self:
        self._cat_mapping = {
            feature: "category"
            for feature in X.columns[X.dtypes == object]  # noqa: E721
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.astype(self._cat_mapping)


class ColumnDropper(TransformerMixin, BaseEstimator):
    def __init__(self, columns: list[str]) -> None:
        self._columns = columns

    def fit(self, X: pd.DataFrame, y: None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self._columns, errors="ignore")


def filter_features(importance: pd.DataFrame, num_features: int) -> list[str]:
    filtered_importance = importance.sort_values("importance", ascending=False).head(
        num_features
    )
    return filtered_importance["feature"].tolist()
