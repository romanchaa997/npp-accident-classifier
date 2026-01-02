# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NPP_", env_file=".env", extra="ignore")

    # Data
    train_path: str = "data/processed/train.parquet"
    val_path: str = "data/processed/val.parquet"
    test_path: str = "data/processed/test.parquet"

    # Model
    window_size: int = 50
    feature_dim: int = 7
    num_classes: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    num_workers: int = 0
    num_epochs: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_tags: float = 1.0

    # Sampler
    use_weighted_sampler: bool = True
    weight_alpha: float = 0.5
    oversample_factor: float = 1.0

    # Checkpoint
    checkpoint_best: str = "models/npp_best.pt"

settings = Settings()
