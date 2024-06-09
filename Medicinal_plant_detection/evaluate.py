import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np

from model import CNN_Model
from data import train_dataset, test_dataset, idx_to_class
from config import ModelConfig

def plot_history(history, metric, filename):
    """
    Function to plot the training and validation metrics over epochs.

    Parameters:
    history (dict): Dictionary containing the history of metrics.
    metric (str): The metric to plot (e.g., 'loss', 'accuracy').
    filename (str): The filename to save the plot.
    """
    os.makedirs('Medicinal_plant_detection/plots', exist_ok=True)

    plt.plot(history[f"train_{metric}"], label=f"train_{metric}")
    plt.plot(history[f"val_{metric}"], label=f"val_{metric}")
    plt.title(metric.capitalize())
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.show()

def visualize_predictions(model, dataset, idx_to_class, device, filename):
    """
    Function to visualize the predictions of the model on a given dataset.

    Parameters:
    model (torch.nn.Module): The trained PyTorch model.
    dataset (torch.utils.data.Dataset): The dataset to visualize predictions for.
    idx_to_class (dict): Dictionary mapping class indices to class names.
    device (torch.device): The device to run the model on (CPU or GPU).
    filename (str): The filename to save the plot of predictions.
    """
    os.makedirs('Medicinal_plant_detection/plots', exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in dataloader:
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

if __name__ == '__main__':
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ModelConfig()
    
    # Initialize and load the model
    model = CNN_Model(config).to(device)
    model = model.load_model(filepath="Medicinal_plant_detection\models\medplantdetec_6m.pth", config=config).to(device)
    model.eval()
    
    # Visualize predictions on training and test datasets
    visualize_predictions(model, train_dataset, idx_to_class, device, "Medicinal_plant_detection/plots/training_predictions.png")
    visualize_predictions(model, test_dataset, idx_to_class, device, "Medicinal_plant_detection/plots/test_predictions.png")