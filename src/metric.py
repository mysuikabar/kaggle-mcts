import numpy as np
from sklearn.metrics import mean_squared_error


def calculate_metrics(
    y: np.ndarray, oof: np.ndarray, fold_assignments: np.ndarray
) -> dict[str, float]:
    assert len(y) == len(oof) == len(fold_assignments)

    fold2rmse = {}
    for fold in sorted(np.unique(fold_assignments)):
        mask = fold_assignments == fold
        rmse = mean_squared_error(y[mask], oof[mask], squared=False)
        fold2rmse[fold] = rmse

    rmse_overall = mean_squared_error(y, oof, squared=False)
    rmse_std = np.std(list(fold2rmse.values()))

    metrics = {
        "rmse": rmse_overall,
        "rmse_std": rmse_std,
        **{f"rmse_fold_{i}": rmse for i, rmse in fold2rmse.items()},
    }

    return metrics
