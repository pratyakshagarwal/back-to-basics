import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix, BinaryF1Score

# Setting device for model training and inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Binary Cross Entropy Loss function
loss_fn = nn.BCELoss()

def bceloss_weighted(pred, target, class_weights):
    """
    Compute the weighted Binary Cross Entropy Loss.

    Parameters:
    pred (torch.Tensor): Predictions from the model.
    target (torch.Tensor): True labels.
    class_weights (torch.Tensor): Weights for the classes.

    Returns:
    torch.Tensor: Weighted BCE loss.
    """
    # Compute standard BCE loss
    loss = loss_fn(pred.view(-1), target)
    # Apply class weights to the loss
    weighted_loss = torch.tensor([class_weights[1].item() if x == 1 else class_weights[0].item() for x in target], dtype=torch.float32).to(device)
    weighted_loss_mean = (loss * weighted_loss).mean()
    return weighted_loss_mean

def plot_history(history, metric, filename):
    """
    Plot training and validation history for a given metric.

    Parameters:
    history (dict): Dictionary containing training history.
    metric (str): Metric to plot (e.g., 'loss', 'accuracy').
    filename (str): Path to save the plot.
    """
    # Create the directory if it doesn't exist
    os.makedirs("CreditCard_Fraud_Detection/plots", exist_ok=True)
    
    # Plot the training and validation history
    plt.plot(history[f"train_{metric}"], label=f"train_{metric}")
    plt.plot(history[f"val_{metric}"], label=f"val_{metric}")
    plt.title(metric.capitalize())
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.show()

def print_report(dataloader, model):
    """
    Print classification report for the model predictions.

    Parameters:
    dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    model (nn.Module): Trained model.
    """
    print("Classification Report:")
    ytest, ypred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)

            ypred.extend(pred.round().cpu().detach().numpy())
            ytest.extend(yb.cpu().detach().numpy())

    print(classification_report(ytest, ypred))

def get_accuracy_with_balanced_data(ds_raw, model, device, threshold=0.3):
    """
    Calculate accuracy and F1 score on balanced test data.

    Parameters:
    ds_raw (pd.DataFrame): Raw dataset.
    model (nn.Module): Trained model.
    device (torch.device): Device to run the model on.
    threshold (float): Threshold for classifying as positive class.

    Returns:
    tuple: Accuracy, F1 score, predictions, and confusion matrix.
    """
    # Create accuracy and F1 score functions
    accuracy_fn = BinaryAccuracy().to(device)
    f1score_fn = BinaryF1Score().to(device)
    confusion_matrix_fn = BinaryConfusionMatrix().to(device)

    # Get classes that contain equal number of both classes
    class_0 = ds_raw[ds_raw['Class'] == 0]
    class_1 = ds_raw[ds_raw['Class'] == 1]

    # Sample 1000 - len(minority_class) from the majority class
    if len(class_1) <= 1000:
        class_0_sample = class_0.sample(n=1000 - len(class_1))
    else:
        raise ValueError("The minority class has more than 1000 examples, adjust the sampling logic")

    # Create a test dataframe
    ds_test = pd.concat([class_1, class_0_sample], ignore_index=True)
    ds_test.reset_index(drop=True, inplace=True)

    print(ds_test['Class'].value_counts())

    # Get features and label
    xb_test = torch.tensor(ds_test.iloc[:, 1:-1].values, dtype=torch.float32).to(device)
    yb_test = torch.tensor(ds_test.iloc[:, -1].values, dtype=torch.float32).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        pred = model(xb_test)

    # Apply threshold after converting to probabilities
    preds = (pred >= threshold).float()

    # Calculate metrics
    test_accuracy = accuracy_fn(preds.view(-1), yb_test.view(-1))
    test_f1 = f1score_fn(preds.view(-1), yb_test.view(-1))
    confusion_matrix = confusion_matrix_fn(preds.view(-1), yb_test.view(-1))

    print(f"AT Threshold: {threshold}")

    return test_accuracy.item(), test_f1.item(), preds.cpu().numpy(), confusion_matrix.cpu().numpy()

def plot_confusion_matrix(cm, class_names, filename):
    """
    Plot and save the confusion matrix.

    Parameters:
    cm (numpy.ndarray): Confusion matrix.
    class_names (list): List of class names.
    filename (str): Path to save the plot.
    """
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.show()
