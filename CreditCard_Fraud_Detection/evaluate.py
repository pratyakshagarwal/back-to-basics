import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

# Importing utility functions and classes
from utils import print_report, get_accuracy_with_balanced_data, plot_confusion_matrix
from data import CreditCardDataset
from model import FraudModel
from config import ModelConfig

if __name__ == '__main__':
    # Load the test dataset and the raw dataset
    test_ds = pd.read_csv(r"CreditCard_Fraud_Detection/datasets/test_ds.csv")
    ds_raw = pd.read_csv(r"CreditCard_Fraud_Detection/datasets/creditcardfraud.csv")

    # Configure device for computation (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model configuration and the pre-trained model
    config = ModelConfig()
    model = FraudModel(config)
    model = model.load_model(filepath=r"CreditCard_Fraud_Detection\fraudetcmodel181k.pth", config=config).to(device)

    # Prepare the test dataset and dataloader
    valdata = CreditCardDataset(test_ds)
    valdataloader = DataLoader(valdata, batch_size=config.batch_size, shuffle=True)

    # Print classification report for the validation dataset
    print_report(valdataloader, model)

    # Evaluate the model with balanced data
    test_acc, test_f1, preds, confusion_matrix = get_accuracy_with_balanced_data(ds_raw, model, device)
    
    # Print evaluation metrics
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    
    # Plot and save the confusion matrix
    plot_confusion_matrix(confusion_matrix, class_names=['Not Fraud', 'Fraud'], filename='CreditCard_Fraud_Detection/plots/cm_train.png')