import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_or_build_tokenizer
from typing import Any
import pandas as pd

def split_train_test(ds_raw, train_ratio=0.8):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    ds_raw (pd.DataFrame): The raw dataset containing question pairs and labels.
    train_ratio (float): The ratio of the dataset to be used for training. The rest will be used for testing.

    Returns:
    train_ds (pd.DataFrame): The training set.
    test_ds (pd.DataFrame): The testing set.
    """
    train_size = int(len(ds_raw) * train_ratio)
    
    train_ds = ds_raw.iloc[:train_size, ]
    test_ds = ds_raw.iloc[train_size:, ]
    
    os.makedirs('datasets', exist_ok=True)
    train_ds.to_csv('Quora_Is_Duplicate_Detection/datasets/train_ds.csv', index=False)
    test_ds.to_csv('Quora_Is_Duplicate_Detection/datasets/test_ds.csv', index=False)
    
    return train_ds, test_ds


class QuoraDataset(Dataset):
    def __init__(self, config, ds_raw: pd.DataFrame, maxlen: int):
        """
        Initializes the QuoraDataset object.

        Parameters:
        config (ModelConfig): Configuration object containing model and training parameters.
        ds_raw (pd.DataFrame): The raw dataset containing question pairs and labels.
        maxlen (int): The maximum length of the tokenized input sequences.
        """
        super().__init__()

        self.ds_raw = ds_raw
        self.questions1 = ds_raw['question1'].astype(str).tolist()
        self.questions2 = ds_raw['question2'].astype(str).tolist()
        self.targets = ds_raw['is_duplicate'].astype(int).tolist()
        self.maxlen = maxlen
        self.tokenizer = get_or_build_tokenizer(config)
        self.pad_token = torch.tensor([self.tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        self.sep_token = torch.tensor([self.tokenizer.token_to_id("[SEP]")], dtype=torch.int64)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
        int: Number of samples in the dataset.
        """
        return len(self.ds_raw)

    def __getitem__(self, index: Any) -> Any:
        """
        Retrieves the input features and target for a given index.

        Parameters:
        index (Any): The index of the sample to retrieve.

        Returns:
        input_features (torch.Tensor): The tokenized and padded input sequence.
        target (torch.Tensor): The target label for the input sequence.
        """
        # Tokenize the questions
        question1_tokens = self.tokenizer.encode(self.questions1[index]).ids
        question2_tokens = self.tokenizer.encode(self.questions2[index]).ids
        target = torch.tensor(self.targets[index], dtype=torch.float32)

        # Combine the tokenized questions with a separator token
        combined_tokens = question1_tokens + [self.sep_token] + question2_tokens
        padlen = self.maxlen - len(combined_tokens)

        # Pad the combined tokens if necessary
        if padlen > 0:
            input_features = combined_tokens + [self.pad_token] * padlen
        else:
            input_features = combined_tokens[:self.maxlen]

        # Convert the input features to a torch tensor
        input_features = torch.tensor(input_features, dtype=torch.int64)

        return input_features, target
