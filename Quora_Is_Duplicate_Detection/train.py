import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
import os 
import numpy as np
import pandas as pd
import warnings

from data import QuoraDataset, split_train_test
from config import ModelConfig
from utils import get_or_build_tokenizer, bceloss_weighted, plot_history
from model import LSTM_Model

# Ignore warnings
warnings.filterwarnings("ignore")

class ModelTrainer():
    def __init__(self, config, device: str) -> None:
        """
        Initialize ModelTrainer.

        Args:
            config (ModelConfig): Configuration object.
            device (str): Device to use for training.
        """
        self.device = device
        self.tokenizer = get_or_build_tokenizer(config)
        self.model = LSTM_Model(config).to(device)
        self.num_params = sum([p.numel() for p in self.model.parameters()])
        print(f"No of Parameters : {self.num_params}")

    def train(self, num_epochs, train_dataset, test_dataset, batch_size, save_checkpoint=True, load_checkpoint=None, verbose=True, save_frequency=1):
        """
        Train the model.

        Args:
            num_epochs (int): Number of epochs to train.
            train_dataset (Dataset): Training dataset.
            test_dataset (Dataset): Validation dataset.
            batch_size (int): Batch size for training.
            save_checkpoint (bool): Whether to save model checkpoints.
            load_checkpoint (str): Path to a saved checkpoint for resuming training.
            verbose (bool): Whether to print training progress.
            save_frequency (int): Frequency of saving checkpoints.

        Returns:
            dict: Training history.
        """
        # DataLoader setup
        self.traindataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valdataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Using {self.device}")
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
        }

        initial_epoch = 1
        if load_checkpoint is not None:
            initial_epoch = self.load_checkpoint(load_checkpoint)

        for epoch in range(initial_epoch, initial_epoch + num_epochs):
            epoch_metrics = {
                'loss': [],
                'accuracy': [],
            }

            # Training loop
            self.model.train()
            for batch in self.traindataloader:
                xb, yb = batch
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                # Forward pass
                pred = self.model(xb)
                
                # Calculate loss
                loss = bceloss_weighted(pred, yb, self.class_weights)
                
                # Backpropagation
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                
                # Metrics calculation
                accuracy = self.accuracy_fn(pred.view(-1), yb)
                
                epoch_metrics['loss'].append(loss.item())
                epoch_metrics['accuracy'].append(accuracy.item())

            # Validation loop
            val_metrics = self.validation_loop(self.valdataloader, self.device)
            
            # Store metrics for history
            history['train_loss'].append(np.mean(epoch_metrics['loss']))
            history['val_loss'].append(np.mean(val_metrics['val_loss']))
            history['train_accuracy'].append(np.mean(epoch_metrics['accuracy']))
            history['val_accuracy'].append(np.mean(val_metrics['val_accuracy']))

            # Print progress
            if verbose:
                print(f"Epoch: {epoch}  |  Train Loss:  {history['train_loss'][-1]:.4f}  |  Train Accuracy: {history['train_accuracy'][-1]:.4f}  |   Val Loss:  {history['val_loss'][-1]:.4f}  |  Val Accuracy: {history['val_accuracy'][-1]:.4f}")
                
            # Save checkpoint
            if epoch % save_frequency == 0 and save_checkpoint:
                self.save_checkpoint(epoch)
                
        # Save final model
        self.model.save_model(f"quoraduplidetec{(self.num_params/1_000_000):.0f}m.pth")
        return history
        
    def validation_loop(self, dataloader, device):
        """
        Run validation loop.

        Args:
            dataloader (DataLoader): Validation dataloader.
            device (str): Device to use.

        Returns:
            dict: Validation metrics.
        """
        self.model.eval()
        val_metrics = {
            'val_loss' : [],
            'val_accuracy' : [],
        }
        
        with torch.no_grad():
            for batch in dataloader:
                val_xb, val_yb = batch
                val_xb, val_yb = val_xb.to(device), val_yb.to(device)
                
                # Get predictions
                pred = self.model(val_xb)
                
                # Calculate loss and accuracy
                val_loss = bceloss_weighted(pred, val_yb, self.class_weights)
                val_accu = self.accuracy_fn(pred.view(-1), val_yb)
                
                val_metrics['val_loss'].append(val_loss.item())
                val_metrics['val_accuracy'].append(val_accu.item())

        self.model.train()
        return val_metrics
    
    def compile(self, loss_fn, optimizer_cls, metric_fn, lr=1e-2, class_weights=None):
        """
        Compile the model with loss function, optimizer, and metric.

        Args:
            loss_fn: Loss function.
            optimizer_cls: Optimizer class.
            metric_fn: Metric function.
            lr (float): Learning rate.
            class_weights (tensor): Class weights.
        """
        if class_weights is None:
            self.class_weights = torch.tensor([1, 1], dtype=torch.float32).to(device)
        else :
            self.class_weights = class_weights
        self.loss_fn = loss_fn
        self.accuracy_fn = metric_fn
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        
    def save_checkpoint(self, epoch):
        """
        Save model checkpoint.

        Args:
            epoch (int): Current epoch number.
        """
        # Create directory if it doesn't exist
        os.makedirs('chkpt', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        filename = f"chkpt/checkpoint_epoch{epoch/100_000_0}.pth"
        torch.save(checkpoint, filename)
        print(f'Checkpoint saved: {filename}')
        
    def load_checkpoint(self, filename):
        """
        Load model checkpoint.

        Args:
            filename (str): Path to the checkpoint file.

        Returns:
            int: New epoch number.
        """
        checkpoint = torch.load(filename)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        new_epoch = checkpoint['epoch']
        
        print(f'Checkpoint loaded: {filename}')
        return new_epoch
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ModelConfig()

    ds_raw = pd.read_csv(r"Quora_Is_Duplicate_Detection\datasets\quora_questions .csv")
    train_ds, test_ds = split_train_test(ds_raw)
    
    loss_fn = nn.BCELoss().to(device)
    accuracy_fn = BinaryAccuracy().to(device)
    
    train_dataset = QuoraDataset(config, train_ds, config.maxlen)
    test_dataset = QuoraDataset(config, test_ds, config.maxlen)

    trainer =  ModelTrainer(config, device)
    trainer.compile(loss_fn=loss_fn, optimizer_cls=torch.optim.Adam, lr=config.lr, metric_fn=accuracy_fn, class_weights=config.class_weights)

    history = trainer.train(num_epochs=config.num_epochs, train_dataset=train_dataset, test_dataset=test_dataset, batch_size=config.batch_size, save_frequency=config.save_frequency)
    
    plot_history(history, 'loss', 'Quora_Is_Duplicate_Detection/plots/loss_plot.png')
    plot_history(history, 'accuracy', 'Quora_Is_Duplicate_Detection/plots/accuracy_plot.png')