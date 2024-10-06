from collections import UserDict
from logging import getLogger
from pathlib import Path

import pandas as pd
import polars as pl

from .base import BaseProcessor

logger = getLogger(__name__)


class FeatureExpressions(UserDict):
    """
    A custom dictionary for storing feature expressions.
    """

    def __setitem__(self, key: str, value: list[pl.Expr]) -> None:
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        if not isinstance(value, list) or not all(
            isinstance(v, pl.Expr) for v in value
        ):
            raise TypeError("Value must be a list of polars expressions")
        self.data[key] = value

    def __getitem__(self, key: str) -> list[pl.Expr]:
        return list(super().__getitem__(key))  # Return a copy to prevent mutation

    def filter(self, feature_names: list[str]) -> "FeatureExpressions":
        """
        Filter the dictionary based on given feature names.
        """
        return FeatureExpressions(
            {k: self.data[k] for k in feature_names if k in self.data}
        )


class FeatureStore:
    """
    A class for storing and retrieving features.
    """

    def __init__(self, dir_path: Path) -> None:
        self._dir_path = dir_path
        self._dir_path.mkdir(parents=True, exist_ok=True)

    def has_feature(self, feature_name: str) -> bool:
        """
        Check if a feature exists in the store.
        """
        file_path = self._dir_path / f"{feature_name}.parquet"
        return file_path.exists()

    def save(self, df: pl.DataFrame, feature_name: str) -> None:
        """
        Save a feature DataFrame to a parquet file.
        """
        file_path = self._dir_path / f"{feature_name}.parquet"
        df.write_parquet(file_path)
        logger.info(f"Feature '{feature_name}' saved to {file_path}")

    def load(self, feature_name: str) -> pl.DataFrame:
        """
        Load a feature DataFrame from a parquet file.
        """
        file_path = self._dir_path / f"{feature_name}.parquet"
        if file_path.exists():
            logger.info(f"Loading feature '{feature_name}' from {file_path}")
            return pl.read_parquet(file_path)
        else:
            logger.info(f"Feature '{feature_name}' not found at {file_path}")
            raise FileNotFoundError(f"Feature {feature_name} not found.")


class FeatureProcessor(BaseProcessor):
    """
    A class for processing features.
    """

    def __init__(
        self,
        feature_expressions: FeatureExpressions,
        feature_store_dir: Path | None = None,
    ) -> None:
        self._feature_expressions = feature_expressions
        self._feature_store = (
            FeatureStore(feature_store_dir) if feature_store_dir else None
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process features on the input DataFrame.
        """
        df_result = pl.DataFrame(df)

        if self._feature_store is None:
            expressions = [
                expr for exprs in self._feature_expressions.values() for expr in exprs
            ]
            return df_result.with_columns(expressions).to_pandas()

        # when feature store is given
        for feature_name, expressions in self._feature_expressions.items():
            try:
                feature = self._feature_store.load(feature_name)
                if len(feature) != len(df_result):
                    raise ValueError(
                        f"Loaded feature '{feature_name}' has {len(feature)} rows, but df_result has {len(df_result)} rows. Row counts must match."
                    )
                df_result = df_result.hstack(feature)
            except FileNotFoundError:
                new_feature = df_result.select(expressions)
                self._feature_store.save(new_feature, feature_name)
                df_result = df_result.hstack(new_feature)

        return df_result.to_pandas()

    def disable_feature_store(self) -> None:
        """
        Disable the feature store.
        """
        self._feature_store = None
