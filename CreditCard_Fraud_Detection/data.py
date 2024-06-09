# Import necessary libraries
import os
import torch
import pandas as pd
from typing import Any
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE

def split_train_test(ds_raw: pd.DataFrame, train_ratio: float = 0.8):
    """
    Splits the raw dataset into training and testing sets based on the specified train_ratio.
    
    Parameters:
    ds_raw (pd.DataFrame): The raw dataset to be split.
    train_ratio (float): The proportion of the dataset to include in the train split. Default is 0.8.
    
    Returns:
    train_ds (pd.DataFrame): The training dataset.
    test_ds (pd.DataFrame): The testing dataset.
    """
    # Calculate the size of the training set
    train_size = int(len(ds_raw) * train_ratio)
    
    # Split the dataset
    train_ds = ds_raw.iloc[:train_size, ]
    test_ds = ds_raw.iloc[train_size:, ]
    
    # Reset index of test set for consistency
    test_ds.reset_index(drop=True, inplace=True)

    # Create directory for saving datasets if it doesn't exist
    os.makedirs('datasets', exist_ok=True)
    
    # Save the train and test datasets to CSV files
    train_ds.to_csv(r'CreditCard_Fraud_Detection\datasets\train_ds.csv', index=False)
    test_ds.to_csv(r'CreditCard_Fraud_Detection\datasets\test_ds.csv', index=False)
    
    # Ensure the split was correct
    assert len(train_ds) + len(test_ds) == len(ds_raw)
    
    return train_ds, test_ds

class CreditCardDataset(Dataset):
    """
    A custom Dataset class for credit card fraud detection.
    This class reads a dataframe, applies SMOTE for handling class imbalance, 
    and prepares the data for model training.
    """

    def __init__(self, ds_raw: pd.DataFrame) -> None:
        """
        Initializes the dataset by extracting features and labels, and applying SMOTE.
        
        Parameters:
        ds_raw (pd.DataFrame): The raw dataframe containing the dataset.
        """
        # Extract features (all columns except the first and the last)
        self.features = ds_raw.iloc[:, 1:-1]
        
        # Extract labels (last column)
        self.labels = ds_raw.iloc[:, -1]
        
        # Store the number of samples
        self.num_samples = len(ds_raw)

        # Apply SMOTE to handle class imbalance
        self.smote = SMOTE(random_state=42)
        self.X_resampled, self.y_resampled = self.smote.fit_resample(self.features, self.labels)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        
        Returns:
        int: Number of samples.
        """
        return self.num_samples

    def __getitem__(self, index: Any) -> torch.Tensor:
        """
        Returns the feature and label tensors for a given index.
        
        Parameters:
        index (Any): The index of the sample to retrieve.
        
        Returns:
        Tuple[torch.Tensor, torch.Tensor]: The feature and label tensors.
        """
        # Convert features and labels to tensors
        x = torch.tensor(self.X_resampled.iloc[index].values, dtype=torch.float32)
        y = torch.tensor(self.y_resampled[index], dtype=torch.float32)
        
        return x, y