import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report
from torchmetrics.classification import BinaryConfusionMatrix, BinaryAccuracy, BinaryF1Score
from model import LSTM_Model

# Read the dataset
ds_raw = pd.read_csv("Quora_Is_Duplicate_Detection\datasets\quora_questions .csv")

# Binary Cross-Entropy Loss function
loss_fn = nn.BCELoss()

# Device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_all_sentence(ds_raw):
    """
    Extracts all sentences from the dataset.

    Parameters:
    ds_raw (DataFrame): Raw dataset containing questions.

    Returns:
    list: List of all sentences.
    """
    corpus = ds_raw['question1'].astype(str).values + ds_raw['question2'].astype(str).values
    return corpus

def get_or_build_tokenizer(config):
    """
    Get or build a WordLevel tokenizer.

    Parameters:
    config (dict): Configuration parameters.

    Returns:
    Tokenizer: WordLevel Tokenizer.
    """
    tokenizer_path = Path(config.tokenizer_file_path)
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SEP]"])
        tokenizer.train_from_iterator(get_all_sentence(ds_raw), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def bceloss_weighted(pred, target, class_weights):
    """
    Weighted Binary Cross-Entropy Loss function.

    Parameters:
    pred (torch.Tensor): Predicted values.
    target (torch.Tensor): Target values.
    class_weights (list): Class weights.

    Returns:
    torch.Tensor: Weighted loss.
    """
    loss = loss_fn(pred.view(-1), target)
    weighted_loss = torch.tensor([class_weights[1].item() if x == 1 else class_weights[0].item() for x in target], dtype=torch.float32).to(device)
    weighted_loss_mean = (loss * weighted_loss).mean()
    return weighted_loss_mean

def plot_history(history, metric, filename):
    """
    Plot training and validation history.

    Parameters:
    history (dict): Training history.
    metric (str): Metric to plot.
    filename (str): File name to save the plot.
    """
    os.makedirs('Quora_Is_Duplicate_Detection/plots', exist_ok=True)
    plt.plot(history[f"train_{metric}"], label=f"train_{metric}")
    plt.plot(history[f"val_{metric}"], label=f"val_{metric}")
    plt.title(metric.capitalize())
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.show()

def plot_roc_curve(y_test, y_pred_prob, filename="plots/roc_plot.png"):
    """
    Plot ROC curve and determine the best threshold.

    Parameters:
    y_test (array-like): True binary labels.
    y_pred_prob (array-like): Predicted probabilities for the positive class.
    filename (str): Path to save the ROC plot.

    Returns:
    float: Best threshold value.
    """
    os.makedirs("Quora_Is_Duplicate_Detection/plots", exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Youden's J statistic
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_threshold = thresholds[best_idx]
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', marker='o', label=f'Best Threshold = {best_threshold:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.show()
    
    return best_threshold

def plot_confusion_matrix(cm, class_names, filename="plots/confusion_matrix.png"):
    """
    Plot the confusion matrix.

    Parameters:
    cm (array-like): Confusion matrix.
    class_names (list): List of class names.
    filename (str): Path to save the confusion matrix plot.
    """
    os.makedirs("Quora_Is_Duplicate_Detection/plots", exist_ok=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.show()

def print_report(dataloader, model, plot_filename="plots/roc_plot.png"):
    """
    Print classification report and plot ROC curve.

    Parameters:
    dataloader (DataLoader): DataLoader for the dataset.
    model (nn.Module): Trained model.
    plot_filename (str): Path to save the ROC plot.
    """
    confusion_matrix_fn = BinaryConfusionMatrix().to(device)
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
    threshold = plot_roc_curve(ytest, ypred, plot_filename)
    print(f"threshold: {threshold}")
    y_pred = torch.tensor(ypred, dtype=torch.float32).view(-1).to(device)
    y_true = torch.tensor(ytest, dtype=torch.float32).to(device)
    confusion_matrix = confusion_matrix_fn(y_pred, y_true)
    return confusion_matrix.cpu().numpy()

class PipeLine:
    def __init__(self, config, model_path):
        """
        Initializes the pipeline object.

        Parameters:
        config (dict): Configuration parameters.
        model_path (str): Path to the saved model.
        """
        self.config = config
        self.tokenizer = get_or_build_tokenizer(config)  # Initialize or load tokenizer
        model = LSTM_Model(config)  # Create LSTM model instance
        self.model = model.load_model(model_path, config).to(device)  # Load pre-trained model
        self.model.eval()  # Set the model to evaluation mode
        self.pad_token = self.tokenizer.token_to_id("[PAD]")  # Get PAD token ID
        self.sep_token = self.tokenizer.token_to_id("[SEP]")  # Get SEP token ID

    def preprocess(self, question1, question2):
        """
        Preprocesses the input questions.

        Parameters:
        question1 (str): First question.
        question2 (str): Second question.

        Returns:
        list: Combined tokens of preprocessed questions.
        """
        question1_tokens = self.tokenizer.encode(question1).ids
        question2_tokens = self.tokenizer.encode(question2).ids
        combined_tokens = question1_tokens + [self.sep_token] + question2_tokens
        return combined_tokens

    def pad_sequences(self, sequences):
        """
        Pads sequences to a fixed length.

        Parameters:
        sequences (list): List of token sequences.

        Returns:
        torch.Tensor: Padded sequences as tensors.
        """
        max_len = min(self.config.maxlen, max(len(seq) for seq in sequences))
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_len:
                seq += [self.pad_token] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            padded_sequences.append(seq)
        return torch.tensor(padded_sequences, dtype=torch.int64)

    def predict(self, questions1, questions2, real=None, threshold=0.5, verbose=True):
        """
        Performs prediction on given pairs of questions.

        Parameters:
        questions1 (str or list): First question or list of questions.
        questions2 (str or list): Second question or list of questions.
        real (list): Real labels (if available).
        threshold (float): Threshold for classification.
        verbose (bool): Whether to print verbose output.

        Returns:
        list: List of predicted results.
        """
        if isinstance(questions1, str):
            questions1 = [questions1]
        if isinstance(questions2, str):
            questions2 = [questions2]

        assert len(questions1) == len(questions2), "The lengths of questions1 and questions2 must match."

        sequences = [self.preprocess(q1, q2) for q1, q2 in zip(questions1, questions2)]
        input_tokens = self.pad_sequences(sequences).to(device)

        with torch.no_grad():
            predictions = self.model(input_tokens).squeeze(1).cpu().numpy()

        results = ['Duplicate' if pred > threshold else 'Not Duplicate' for pred in predictions]

        if verbose:
            for i, (q1, q2, result) in enumerate(zip(questions1, questions2, results)):
                print("---------------------------------------------------------------------------------------------")
                print(f"Question 1: {q1}")
                print(f"Question 2: {q2}")
                if real is not None:
                    print(f"Real: {['Not Duplicate', 'Duplicate'][real[i]]}")
                print(f"Output: {result}")
                print("---------------------------------------------------------------------------------------------")

        return results
    
def get_accuracy_with_threshold(config, ds, model_path, device, threshold=0.5, sample_size=5000):
    """
    Get accuracy and F1 score with a specific threshold.

    Parameters:
    config (dict): Configuration dictionary for the model.
    ds (DataFrame): Dataset containing questions and labels.
    model_path (str): Path to the saved model.
    device (torch.device): Device to run the model on.
    threshold (float): Threshold for classification.
    sample_size (int): Number of samples to use from each class.

    Returns:
    tuple: Accuracy, F1 score, and confusion matrix.
    """
    accuracy_fn = BinaryAccuracy().to(device)
    f1score_fn = BinaryF1Score().to(device)
    confusion_matrix_fn = BinaryConfusionMatrix().to(device)
    
    # Sample data from each class
    class_0 = ds[ds['is_duplicate'] == 0]
    class_1 = ds[ds['is_duplicate'] == 1]
    class_0_sample = class_0.sample(n=sample_size)
    class_1_sample = class_1.sample(n=sample_size)
    
    # Concatenate sampled data
    ds_test = pd.concat([class_1_sample, class_0_sample], ignore_index=True)
    ds_test.reset_index(drop=True, inplace=True)
    
    # Extract questions and labels
    questions1 = ds_test['question1'].astype(str).tolist()
    questions2 = ds_test['question2'].astype(str).tolist()
    targets = ds_test['is_duplicate'].astype(int).tolist()
    
    # Initialize pipeline with the model
    pipeline = PipeLine(config, model_path)
    
    # Get predictions
    preds = pipeline.predict(questions1, questions2, threshold=threshold, verbose=False)
    
    # Convert predictions to tensors
    preds = torch.tensor([1 if pred == 'Duplicate' else 0 for pred in preds], dtype=torch.float32).to(device)
    targets = torch.tensor(targets, dtype=torch.float32).to(device)
    
    # Calculate accuracy, F1 score, and confusion matrix
    accuracy = accuracy_fn(preds, targets)
    f1score = f1score_fn(preds, targets)
    confusion_matrix = confusion_matrix_fn(preds, targets)
    
    print(f"At Threshold {threshold}")
    print(f"Accuracy {accuracy} \n F1Score {f1score}")
    
    return accuracy, f1score, confusion_matrix.cpu().numpy()
