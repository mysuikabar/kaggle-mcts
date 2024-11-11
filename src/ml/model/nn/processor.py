import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from typing_extensions import Self


class Preprocessor:
    """
    Preprocessor for tabular data.

    - Standardize numerical features
    - Encode categorical features with LabelEncoder
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._label_encoders: dict[str, OrdinalEncoder] | None = None

    def fit(self, X: pd.DataFrame, categorical_feature_dims: dict[str, int]) -> Self:
        self._numerical_features = [
            col for col in X.columns if col not in categorical_feature_dims
        ]

        self._scaler = StandardScaler().fit(X[self._numerical_features])
        self._label_encoders = {}
        for feature, dim in categorical_feature_dims.items():
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=dim
            )  # encode unseen values as a largest integer
            self._label_encoders[feature] = encoder.fit(
                X[feature].values.reshape(-1, 1)
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        X_copy[self._numerical_features] = X_copy[self._numerical_features].astype(
            float
        )
        X_copy.loc[:, self._numerical_features] = self._scaler.transform(  # type: ignore
            X_copy[self._numerical_features]
        )

        for feature, encoder in self._label_encoders.items():  # type: ignore
            X_copy[feature] = encoder.transform(X_copy[feature].values.reshape(-1, 1))

        return X_copy

    def fit_transform(
        self, X: pd.DataFrame, categorical_feature_dims: dict[str, int]
    ) -> pd.DataFrame:
        return self.fit(X, categorical_feature_dims).transform(X)
