from ml.model.gbdt import CatBoostConfig, LightGBMConfig, XGBoostConfig

lightgbm_config = LightGBMConfig(
    objective="regression",
    metric="rmse",
    boosting_type="gbdt",
    num_leaves=30,
    learning_rate=0.05,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    num_boost_round=10000,
    early_stopping_rounds=50,
    max_bin=255,
    device="gpu",
)

xgboost_config = XGBoostConfig(
    objective="reg:squarederror",
    eval_metric="rmse",
    booster="gbtree",
    learning_rate=0.05,
    min_split_loss=0,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    reg_lambda=1,
    reg_alpha=1,
    colsample_bytree=1,
    colsample_bylevel=1,
    num_boost_round=10000,
    early_stopping_rounds=50,
    device="gpu",
)

catboost_config = CatBoostConfig(
    loss_function="RMSE",
    eval_metric="RMSE",
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=1,
    random_strength=1,
    bagging_temperature=1,
    od_type="Iter",
    od_wait=10,
    iterations=10000,
    early_stopping_rounds=50,
    task_type="GPU",
)
