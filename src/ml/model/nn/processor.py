import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing_extensions import Self


class Preprocessor:
    """
    Preprocessor for tabular data.

    - Standardize numerical features
    - Encode categorical features with LabelEncoder
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._label_encoders: dict[str, LabelEncoder] | None = None

    def fit(self, X: pd.DataFrame, categorical_feature_dims: dict[str, int]) -> Self:
        self._categorical_features = list(categorical_feature_dims.keys())
        self._numerical_features = [
            col for col in X.columns if col not in self._categorical_features
        ]

        self._scaler = StandardScaler().fit(X[self._numerical_features])
        self._label_encoders = {}
        for feature in self._categorical_features:
            unique_values = X[feature].unique()
            unique_values.append("<UNK>")
            self._label_encoders[feature] = LabelEncoder().fit(unique_values)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        X_copy.loc[:, self._numerical_features] = self._scaler.transform(  # type: ignore
            X_copy[self._numerical_features]
        )

        for feature, encoder in self._label_encoders.items():  # type: ignore
            known_values = set(encoder.classes_)
            X_copy[feature] = X_copy[feature].apply(
                lambda x: x if x in known_values else "<UNK>"
            )
            X_copy[feature] = encoder.transform(X_copy[feature])

        return X_copy

    def fit_transform(
        self, X: pd.DataFrame, categorical_feature_dims: dict[str, int]
    ) -> pd.DataFrame:
        return self.fit(X, categorical_feature_dims).transform(X)
