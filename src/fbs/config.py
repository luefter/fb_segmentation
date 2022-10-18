from dataclasses import dataclass


@dataclass
class ModelParams:
    number_of_channels: int
    number_of_classes: int


@dataclass
class Training:
    learning_rate: float
    number_of_epochs: int


@dataclass
class Path:
    model_weights: str
    dataset: str


@dataclass
class Config:
    path: Path
    training: Training
    params: ModelParams


params = ModelParams(number_of_channels=3, number_of_classes=2)
training = Training(learning_rate=0.001, number_of_epochs=5)
path = Path(model_weights="src/fbs/data/model_weights/model", dataset="src/fbs/data")

config = Config(params=params, training=training, path=path)
