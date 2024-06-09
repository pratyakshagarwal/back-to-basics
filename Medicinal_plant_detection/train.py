import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics import Accuracy
import numpy as np

from evaluate import plot_history  # Importing the plot_history function from the evaluate module
from config import ModelConfig  # Importing ModelConfig class from config module
from data import test_dataset, train_dataset, idx_to_class  # Importing datasets and dictionaries from data module
from model import CNN_Model  # Importing CNN_Model class from model module

class Trainer:
    def __init__(self, model, config: ModelConfig, device) -> None:
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # Initializing Adam optimizer
        self.criterion = nn.NLLLoss().to(device)  # Initializing negative log likelihood loss
        self.config = config
        self.device = device
        self.accuracy_fn = Accuracy(task="multiclass", num_classes=40).to(device)  # Initializing accuracy metric
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=6, gamma=0.6)  # Learning rate scheduler

    def run(self, epochs, train_dataset, val_dataset=None):
        print(f"using {self.device}")

        traindataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)  # DataLoader for training set
        if val_dataset is not None:
            valdataloader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)  # DataLoader for validation set

        # Container to store metrics value during training
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

        for epoch in range(1, epochs + 1):
            epoch_metrics = {
                'loss': [],
                'accuracy': []
            }

            for batch in traindataloader:
                xb, yb = batch
                xb, yb = xb.to(self.device), yb.to(self.device)

                # Get predictions
                pred = self.model(xb)

                # Get loss
                loss = self.criterion(pred, yb)

                # Backpropagation
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                epoch_metrics['loss'].append(loss.item())
                epoch_metrics['accuracy'].append(self.get_accuracy(pred, yb).item())

            # Validation step
            val_metrics = self.validation_loop(valdataloader)

            # Update history
            history['train_loss'].append(np.mean(epoch_metrics['loss']))
            history['val_loss'].append(np.mean(val_metrics['val_loss']))
            history['train_accuracy'].append(np.mean(epoch_metrics['accuracy']))
            history['val_accuracy'].append(np.mean(val_metrics['val_accuracy']))

            print(f"Epoch: {epoch}  |  Train Loss: {history['train_loss'][epoch-1]:.4f} | Val Loss: {history['val_loss'][epoch-1]:.4f}  |  Train Accuracy: {history['train_accuracy'][epoch-1]:.4f}  |  Val Accuracy: {history['val_accuracy'][epoch-1]:.4f} ")
            
            # Step the scheduler
            self.scheduler.step()
            
        return history

    def validation_loop(self, valdataloader):
        self.model.eval()

        val_metrics = {
            'val_loss': [],
            'val_accuracy': []
        }
        for batch in valdataloader:
            val_xb, val_yb = batch
            val_xb, val_yb = val_xb.to(self.device), val_yb.to(self.device)

            # Get prediction
            val_pred = self.model(val_xb)

            # Get loss
            val_loss = self.criterion(val_pred, val_yb)

            # Calculate accuracy
            val_accuracy = self.get_accuracy(val_pred, val_yb)

            val_metrics['val_loss'].append(val_loss.item())
            val_metrics['val_accuracy'].append(val_accuracy.item())

        self.model.train()
        return val_metrics

    def get_accuracy(self, pred_probs, yb):
        pred = torch.argmax(pred_probs, dim=-1)
        accuracy = self.accuracy_fn(pred, yb)
        return accuracy
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ModelConfig()
    model = CNN_Model(config).to(device)
    print(f"No of Paramters {sum([p.numel() for p in model.parameters()])}")

    model_trainer = Trainer(model, config, device)
    history = model_trainer.run(40, train_dataset, test_dataset)
    model_trainer.model.save_model("models/medplantdetec_6m.pth")  # Saving the trained model

    plot_history(history, "loss", "Medicinal_plant_detection/plots/loss_plot.png")  # Plotting the loss history
    plot_history(history, "accuracy", "Medicinal_plant_detection/plots/accuracy_plot.png")  # Plotting the accuracy history