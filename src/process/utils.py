import numpy as np
import pandas as pd
import polars as pl
from typing_extensions import Self


def constant_columns(df: pl.DataFrame) -> list[str]:
    return list(
        np.array(df.columns)[df.select(pl.all().n_unique() == 1).to_numpy().ravel()]
    )


class CategoricalConverter:
    def __init__(self) -> None:
        self._cat_mapping: dict[str, pd.CategoricalDtype] | None = None

    def fit(self, df: pd.DataFrame) -> Self:
        self._cat_mapping = {
            feature: "category"
            for feature in df.columns[df.dtypes == object]  # noqa: E721
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.astype(self._cat_mapping)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
