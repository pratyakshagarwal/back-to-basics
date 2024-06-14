import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from torch.utils.data import DataLoader

def plot_history(history, metric, filename):
    """
    Plot training history metrics (e.g., loss, accuracy) over epochs.

    Args:
    - history (dict): Dictionary containing training and validation metrics over epochs.
                      Expected keys: 'train_metric', 'val_metric'
    - metric (str): Metric to plot (e.g., 'loss', 'accuracy')
    - filename (str): File name to save the plot (including file extension)

    Returns:
    None
    """
    plt.plot(history[f"train_{metric}"], label=f"train_{metric}")
    plt.plot(history[f"val_{metric}"], label=f"val_{metric}")
    plt.title(metric.capitalize())
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.show()

class CosineLrScheduler():
    """
    Cosine Learning Rate Scheduler with Warmup.

    Args:
    - warmup_steps (int): Number of warmup steps.
    - max_lr (float): Maximum learning rate after warmup.
    - min_lr (float): Minimum learning rate.
    - max_steps (int): Total number of training steps.

    Methods:
    - get_lr(epoch): Returns the learning rate for a given epoch.
    """
    def __init__(self, warmup_steps=15, max_lr=6e-3, min_lr=3e-4, max_steps=30):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_steps = max_steps

    def get_lr(self, epoch):
        """
        Returns the learning rate for a given epoch.

        Args:
        - epoch (int): Current epoch number.

        Returns:
        - lr (float): Learning rate for the epoch.
        """
        if epoch < self.warmup_steps:
            return self.max_lr * (epoch + 1) / self.warmup_steps
        if epoch > self.warmup_steps:
            return self.min_lr
        decay_ratio = (epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

def visualize_predictions(model, dataset, idx_to_class, device, filename):
    """
    Visualize model predictions on a sample batch from a dataset.

    Args:
    - model (torch.nn.Module): Trained PyTorch model.
    - dataset (torch.utils.data.Dataset): Dataset to visualize.
    - idx_to_class (dict): Dictionary mapping class indices to class names.
    - device (torch.device): Device (CPU or GPU) to run inference on.
    - filename (str): File name to save the visualization plot.

    Returns:
    None
    """
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for batch in dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # Get predictions
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Move to CPU and convert to numpy
            images = images.cpu().numpy()
            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()

            # Plot images with predictions
            fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
            for i in range(len(images)):
                img = np.transpose(images[i], (1, 2, 0))  # Convert to HWC format
                img = img * 0.5 + 0.5  # Unnormalize the image if it was normalized
                label = labels[i]
                pred = preds[i]
                real_class_name = idx_to_class[label]
                predicted_class_name = idx_to_class[pred]

                ax = axes[i] if len(images) > 1 else axes
                ax.imshow(img)
                ax.set_title(f'Pred: {predicted_class_name}\nReal: {real_class_name}')
                ax.axis('off')

            plt.savefig(filename)
            plt.show()
            break  # Only visualize one batch for simplicity
