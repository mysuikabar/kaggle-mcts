from dataclasses import dataclass, field


@dataclass
class NNConfig:
    embedding_dim: int = 4
    hidden_dims: list[int] = field(default_factory=lambda: [1024, 512, 256])
    dropout_rate: float = 0.1
    learning_rate: float = 1e-4
    scheduler_patience: int = 5
    max_epochs: int = 1000
    early_stopping_patience: int = 10
    batch_size: int = 512
