from dataclasses import dataclass

@dataclass
class ModelConfig:
  in_features = 29
  out_features = 1
  hidden_dim = 128
  dropout = 0.0
  batch_size = 128
  lr = 1e-2