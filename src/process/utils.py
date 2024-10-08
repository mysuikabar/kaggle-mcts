import pandas as pd
from typing_extensions import Self


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
