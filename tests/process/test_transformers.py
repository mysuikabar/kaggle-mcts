import pandas as pd

from process.transformers import TabularDataTransformer


def test_tabular_data_transformer():  # type: ignore
    X_tr = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0],
            "num2": [0.1, 0.2, 0.3, 0.4],
            "cat1": ["A", "B", "A", "C"],
            "cat2": ["X", "Y", "X", "Z"],
        }
    )

    X_va = pd.DataFrame(
        {
            "num1": [5.0, 6.0, 7.0, 8.0],
            "num2": [0.5, 0.6, 0.7, 0.8],
            "cat1": ["B", "C", "B", "A"],
            "cat2": ["W", "Z", "Y", "X"],
        }
    )

    transformer = TabularDataTransformer()
    transformer.fit(X_tr)
    transformed = transformer.transform(X_va)

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.index.equals(X_va.index)
    assert list(transformed.columns) == list(X_va.columns)
    assert transformer.n_categories_ == {"cat1": 4, "cat2": 4}
    assert transformed["cat1"].min() == 1
    assert transformed["cat2"].min() == 0
