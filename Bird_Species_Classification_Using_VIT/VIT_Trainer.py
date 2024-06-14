import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
from vit import VIT  # Assuming VIT is the Vision Transformer model
from utils import CosineLrScheduler  # Custom learning rate scheduler

warnings.filterwarnings('ignore')

class VIT_Trainer:
    """
    Trainer class for training Vision Transformer (VIT) model.
    
    Args:
        config (object): Configuration object containing model and training hyperparameters.
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
    """
    
    def __init__(self, config, device):
        """
        Initializes the VIT_Trainer object.
        
        Args:
            config (object): Configuration object containing model and training hyperparameters.
            device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
        """
        # Create directory for TensorBoard logs
        os.makedirs(r"runs\vtmodel", exist_ok=True)
        
        # Initialize the Vision Transformer model
        self.model = VIT(config).to(device)
        self.config = config
        self.device = device
        self.writer = SummaryWriter(log_dir=config.log_dir)  # TensorBoard writer
        
        # Count the number of model parameters
        self.num_params = sum([p.numel() for p in self.model.parameters()])
        print(f"No of Parameters: {self.num_params}")

    def validation_loop(self, valdataloader, writer=None, val_global_step=0):
        """
        Runs validation loop for the VIT model.
        
        Args:
            valdataloader (DataLoader): DataLoader for the validation dataset.
            writer (SummaryWriter, optional): TensorBoard writer for logging validation metrics. Default is None.
            val_global_step (int, optional): Global step count for TensorBoard logging. Default is 0.
        
        Returns:
            dict: Validation metrics including validation loss and accuracy.
        """
        self.model.eval()  # Set model to evaluation mode
        val_metrics = {
            'val_loss': [],
            'val_accuracy': []
        }
        
        for batch in valdataloader:
            val_xb, val_yb = batch
            val_xb, val_yb = val_xb.to(self.device), val_yb.to(self.device)
            
            with torch.no_grad():
                val_logits = self.model(val_xb)
            
            val_loss = F.cross_entropy(val_logits, val_yb)  # Calculate validation loss
            val_accuracy = self.get_accuracy(val_logits, val_yb)  # Calculate validation accuracy
            
            val_metrics['val_loss'].append(val_loss.item())
            val_metrics['val_accuracy'].append(val_accuracy.item())
            
            if writer:
                writer.add_scalar('val loss', val_loss.item(), val_global_step)
                writer.add_scalar('val accuracy', val_accuracy.item(), val_global_step)
                val_global_step += 1
            
        self.model.train()  # Set model back to training mode
        return val_metrics
    
    def run(self, num_epochs, train_dataset, val_dataset, verbose=True, load_checkpoint=None, save_checkpoint=True, lr_scheduler=None):
        """
        Runs the training loop for the VIT model.
        
        Args:
            num_epochs (int): Number of epochs to train the model.
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset.
            verbose (bool, optional): Whether to print training progress. Default is True.
            load_checkpoint (str, optional): Path to a checkpoint file to resume training. Default is None.
            save_checkpoint (bool, optional): Whether to save checkpoints during training. Default is True.
            lr_scheduler (CosineLrScheduler, optional): Learning rate scheduler object. Default is None.
        
        Returns:
            dict: History object containing training metrics (train_loss, train_accuracy, val_loss, val_accuracy).
        """
        print(f"Using device: {self.device}")
        
        if lr_scheduler is None:
            lr_scheduler = CosineLrScheduler()  # Default learning rate scheduler
        
        # Create DataLoader instances for training and validation datasets
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4)

        initial_epoch = 0
        global_step = 0
        val_global_step = 0
        
        # Load checkpoint if specified
        if load_checkpoint:
            initial_epoch = self.load_checkpoint(load_checkpoint)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
        }

        # Main training loop
        for epoch in range(initial_epoch, initial_epoch + num_epochs):
            epoch_metrics = {
                'train_loss': [],
                'train_accuracy': [],
            }

            # Iterate over batches in the training DataLoader
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}/{initial_epoch + num_epochs}"):
                xb, yb = batch
                xb, yb = xb.to(self.device), yb.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)  # Zero gradients

                logits = self.model(xb)  # Forward pass
                loss = F.cross_entropy(logits, yb)  # Calculate loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Optimizer step

                accuracy = self.get_accuracy(logits, yb)  # Calculate accuracy

                # Log metrics to TensorBoard
                self.writer.add_scalar('train loss', loss.item(), global_step)
                self.writer.add_scalar('train accuracy', accuracy.item(), global_step)

                epoch_metrics['train_loss'].append(loss.item())
                epoch_metrics['train_accuracy'].append(accuracy.item())

                global_step += 1

            # Run validation loop
            val_metrics = self.validation_loop(val_dataloader, writer=self.writer, val_global_step=val_global_step)

            # Update history with epoch metrics
            history['train_loss'].append(np.mean(epoch_metrics['train_loss']))
            history['train_accuracy'].append(np.mean(epoch_metrics['train_accuracy']))
            history['val_loss'].append(np.mean(val_metrics['val_loss']))
            history['val_accuracy'].append(np.mean(val_metrics['val_accuracy']))

            # Save checkpoint if specified
            if save_checkpoint and epoch % self.config.save_frequency == 0:
                self.save_checkpoint(epoch)

            # Print epoch metrics if verbose
            if verbose:
                print(f"Epoch {epoch}  |  Train Loss {history['train_loss'][-1]:.4f}  |  Train Accuracy {history['train_accuracy'][-1]:.4f}  |  Val Loss {history['val_loss'][-1]:.4f}  |  Val Accuracy {history['val_accuracy'][-1]:.4f}")

            # Update learning rate using scheduler
            lr = lr_scheduler.get_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # Save the final model
        self.model.save_model(filename=f"vitmodel{(self.num_params/1_000_000):.0f}m.pth")
        
        return history
    
    def compile(self, optimizer_cls, optimizer_params, accuracy_fn):
        """
        Compiles the trainer with optimizer and accuracy function.
        
        Args:
            optimizer_cls (torch.optim.Optimizer): Optimizer class (e.g., AdamW).
            optimizer_params (dict): Parameters for the optimizer.
            accuracy_fn (torchmetrics.Metric): Accuracy metric function.
        """
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        self.accuracy_fn = accuracy_fn.to(self.device)  # Move accuracy function to device

    def get_accuracy(self, pred_probs, yb):
        """
        Calculates accuracy given predicted probabilities and true labels.
        
        Args:
            pred_probs (torch.Tensor): Predicted probabilities from the model.
            yb (torch.Tensor): True labels.
        
        Returns:
            torch.Tensor: Accuracy value.
        """
        pred = torch.argmax(pred_probs, dim=-1)  # Get predicted labels
        accuracy = self.accuracy_fn(pred, yb)  # Calculate accuracy
        return accuracy

    def save_checkpoint(self, epoch):
        """
        Saves model and optimizer state as a checkpoint.
        
        Args:
            epoch (int): Current epoch number.
        """
        os.makedirs('chkpt', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        filename = os.path.join('chkpt', f"{self.config.checkpoint_path}{epoch}.pth")
        torch.save(checkpoint, filename)
        print(f'Checkpoint saved: {filename}')
        
    def load_checkpoint(self, filename):
        """
        Loads model and optimizer state from a checkpoint file.
        
        Args:
            filename (str): Path to the checkpoint file.
        
        Returns:
            int: Epoch number from the checkpoint.
        """
        checkpoint = torch.load(filename)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        
        print(f'Checkpoint loaded: {filename}')
        return epoch
    

if __name__ == '__main__':
    pass
