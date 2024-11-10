from ml.model.nn import NNConfig

nn_config = NNConfig(
    categorical_feature_dims={
        "p1_selection": 4,
        "p1_playout": 3,
        "p1_bounds": 2,
        "p2_selection": 4,
        "p2_playout": 3,
        "p2_bounds": 2,
        "LudRules_players": 10,
    },
    embedding_dim=4,
    hidden_dims=[512, 256, 128, 64],
    dropout_rate=0.1,
    learning_rate=1e-4,
    early_stopping_patience=5,
    max_epochs=1000,
    batch_size=1024,
)
