from collections import UserDict
from pathlib import Path

import polars as pl

from consts import REPO_ROOT


class FeatureExpressions(UserDict):
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
        return FeatureExpressions(
            {k: self.data[k] for k in feature_names if k in self.data}
        )


class FeatureStore:
    def __init__(self, dir_path: Path = REPO_ROOT / "data" / "feature_store") -> None:
        self._dir_path = dir_path / "features"
        self._dir_path.mkdir(parents=True, exist_ok=True)

    def save(self, df: pl.DataFrame, feature_name: str) -> None:
        file_path = self._dir_path / f"{feature_name}.parquet"
        df.write_parquet(file_path)

    def load(self, feature_name: str) -> pl.DataFrame:
        file_path = self._dir_path / f"{feature_name}.parquet"
        if file_path.exists():
            return pl.read_parquet(file_path)
        else:
            raise FileNotFoundError(f"Feature {feature_name} not found.")


class FeatureProcessor:
    def __init__(self, feature_store: FeatureStore | None = None) -> None:
        self._feature_store = feature_store

    def run(
        self, df: pl.DataFrame, feature_expressions: FeatureExpressions
    ) -> pl.DataFrame:
        df_result = df.clone()

        if self._feature_store is None:
            expressions = [
                expr for exprs in feature_expressions.values() for expr in exprs
            ]
            return df_result.with_columns(expressions)

        # when feature store is given
        for feature_name, expressions in feature_expressions.items():
            try:
                feature = self._feature_store.load(feature_name)
                df_result = df_result.hstack(feature)
            except FileNotFoundError:
                new_feature = df_result.select(expressions)
                self._feature_store.save(new_feature, feature_name)
                df_result = df_result.hstack(new_feature)

        return df_result
