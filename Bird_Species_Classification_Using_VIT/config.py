from dataclasses import dataclass

@dataclass
class VITConfig:
    img_size: int = 224
    num_channels: int = 3
    patch_size: int = 16
    embedding_dim: int = 256
    dropout: float = 0.1
    mlp_size: int = 1024
    nlayer: int = 8
    nheads: int = 8
    num_classes: int = 525
    batch_size = 64
    log_dir = r"runs\vtmodel"
    checkpoint_path  = "checkpoint_epoch"
    save_frequency = 5