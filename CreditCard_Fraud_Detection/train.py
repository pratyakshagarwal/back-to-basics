import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score
import pandas as pd
import numpy as np

# Importing custom modules
from data import CreditCardDataset, split_train_test
from utils import bceloss_weighted, plot_history
from model import FraudModel
from config import ModelConfig

def val_loop(model, dataloader, class_weights, f1score_fn, device):
    """
    Perform validation loop over the validation dataset.

    Parameters:
    - model (nn.Module): The neural network model to evaluate.
    - dataloader (DataLoader): DataLoader for the validation data.
    - class_weights (torch.Tensor): Weights for the classes.
    - f1score_fn (BinaryF1Score): F1 Score metric function.
    - device (torch.device): Device to perform computations on (CPU/GPU).

    Returns:
    - val_metrics (dict): Dictionary containing validation loss and F1 score.
    """
    model.eval()

    val_metrics = {
        'val_loss': [],
        'val_f1score': []
    }

    for batch in dataloader:
        val_xb, val_yb = batch
        val_xb, val_yb = val_xb.to(device), val_yb.to(device)

        # Get predictions
        pred = model(val_xb)

        # Calculate loss and F1 score
        val_loss = bceloss_weighted(pred, val_yb, class_weights)
        val_f1score = f1score_fn(pred.view(-1), val_yb)

        # Append values to the metrics container
        val_metrics['val_loss'].append(val_loss.item())
        val_metrics['val_f1score'].append(val_f1score.item())

    model.train()
    return val_metrics

def train(model, traindataloader, valdataloader, optimizer, f1score_fn, class_weights, device):
    """
    Train the neural network model.

    Parameters:
    - model (nn.Module): The neural network model to train.
    - traindataloader (DataLoader): DataLoader for the training data.
    - valdataloader (DataLoader): DataLoader for the validation data.
    - optimizer (torch.optim.Optimizer): Optimizer for the model.
    - f1score_fn (BinaryF1Score): F1 Score metric function.
    - class_weights (torch.Tensor): Weights for the classes.
    - device (torch.device): Device to perform computations on (CPU/GPU).

    Returns:
    - history (dict): Dictionary containing the training and validation history.
    """
    print(f"Using {device}")

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1score': [],
        'val_f1score': []
    }

    for epoch in range(1, 11):
        epoch_metrics = {
            'train_loss': [],
            'train_f1score': []
        }

        for batch in traindataloader:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)

            # Calculate the loss
            pred = model(xb)
            loss = bceloss_weighted(pred, yb, class_weights)

            # Train the model
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_metrics['train_loss'].append(loss.item())
            epoch_metrics['train_f1score'].append(f1score_fn(pred.view(-1), yb).item())

        # Validate the model
        val_metrics = val_loop(model, valdataloader, class_weights, f1score_fn, device)

        history['train_loss'].append(np.mean(epoch_metrics['train_loss']))
        history['val_loss'].append(np.mean(val_metrics['val_loss']))
        history['train_f1score'].append(np.mean(epoch_metrics['train_f1score']))
        history['val_f1score'].append(np.mean(val_metrics['val_f1score']))

        print(f"Epoch: {epoch}  | Train loss: {history['train_loss'][epoch-1]:.4f}  | Val loss: {history['val_loss'][epoch-1]:.4f}  |  Train F1score: {history['train_f1score'][epoch-1]:.4f}  |  Val F1score {history['val_f1score'][epoch-1]:.4f}")
    
    return history

if __name__ == '__main__':
    # Load the dataset
    ds_raw = pd.read_csv("CreditCard_Fraud_Detection/datasets/creditcardfraud.csv")
    train_ds, val_ds = split_train_test(ds_raw=ds_raw)
    
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = ModelConfig()
    model = FraudModel(config).to(device)
    print(summary(model=model, input_size=(config.batch_size, config.in_features)))

    # Initialize the class weights
    class_weights = torch.tensor([1, 8], dtype=torch.float32).to(device)

    # Initialize the loss function and F1 score metric
    loss_fn = nn.BCELoss(reduction='none').to(device)
    f1score_fn = BinaryF1Score().to(device)

    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Initialize the datasets and dataloaders
    train_dataset = CreditCardDataset(train_ds)
    traindataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = CreditCardDataset(val_ds)
    valdataloader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=True)

    # Train the model
    history = train(model, traindataloader, valdataloader, optimizer, f1score_fn, class_weights, device)
    model.save_model('fraud_detection_model.pth')

    # Plot training history
    plot_history(history, 'loss', 'CreditCard_Fraud_Detection/plots/loss_plot.png')
    plot_history(history, 'f1score', 'CreditCard_Fraud_Detection/plots/f1score_plot.png')
