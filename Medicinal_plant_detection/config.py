from dataclasses import dataclass
from data import idx_to_class

@dataclass
class ModelConfig():
  img_size = 224
  kernel_size = 3
  in_channels = 3
  out_channels = 8
  pool_kernel_size = 2
  pool_stride =  2
  num_classes = len(idx_to_class)
  hidden_dim = 128
  stride = 1
  padding = 1
  dropout = 0.25
  batch_size = 64
  lr = 1e-3