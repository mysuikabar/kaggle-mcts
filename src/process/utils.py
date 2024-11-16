import pandas as pd


def filter_features(importance: pd.DataFrame, num_features: int) -> list[str]:
    filtered_importance = importance.sort_values("importance", ascending=False).head(
        num_features
    )
    return filtered_importance["feature"].tolist()
