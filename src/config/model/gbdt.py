from ml.model.gbdt import LightGBMConfig

lightgbm_config = LightGBMConfig(
    objective="regression",
    metric="rmse",
    boosting_type="gbdt",
    num_leaves=30,
    learning_rate=0.05,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    num_boost_round=1000,
    early_stopping_rounds=10,
    device="gpu",
)
