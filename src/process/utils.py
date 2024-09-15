import numpy as np
import pandas as pd
import polars as pl


def constant_columns(df: pl.DataFrame) -> list[str]:
    return list(
        np.array(df.columns)[df.select(pl.all().n_unique() == 1).to_numpy().ravel()]
    )


class CategoricalConverter:
    def __init__(self) -> None:
        self._cat_mapping: dict[str, pd.CategoricalDtype] | None = None

    def fit(self, df: pd.DataFrame) -> None:
        self._cat_mapping = {
            feature: pd.CategoricalDtype(categories=list(set(df[feature])))
            for feature in df.columns[df.dtypes == object]  # noqa: E721
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.astype(self._cat_mapping)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
