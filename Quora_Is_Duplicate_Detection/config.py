import pandas as pd
from dataclasses import dataclass
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds_raw = pd.read_csv("Quora_Is_Duplicate_Detection\datasets\quora_questions .csv")

@dataclass
class ModelConfig():
  tokenizer_file_path = "tokenizer_file.json"
  maxlen =  max([len(x.split(' ')) for x in ds_raw['question1'].astype(str).values + ds_raw['question2'].astype(str).values])
  embd_dim = 128
  input_vocab = 30000
  hidden_dim = 48
  n_layers = 2
  dropout = 0.1
  bidirectional = True
  lr = 1e-3
  batch_size = 64
  save_frequency = 5
  num_epochs = 20
  # Initialize the weight tensor / weight given to class 0 --> 1 and class 1 --> 3
  class_weights = torch.tensor([1, 5], dtype=torch.float32).to(device)