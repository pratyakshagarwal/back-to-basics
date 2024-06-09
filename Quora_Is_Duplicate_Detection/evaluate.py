import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Importing custom modules
from model import LSTM_Model
from config import ModelConfig
from data import QuoraDataset
from utils import print_report, plot_confusion_matrix, get_accuracy_with_threshold, PipeLine

if __name__ == '__main__':
    # Setting the device to CUDA if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading model configuration
    config = ModelConfig()

    # Loading the pre-trained model
    model = LSTM_Model(config)
    model = model.load_model(r"Quora_Is_Duplicate_Detection\quoraduplidetec4m.pth", config).to(device)
    model.eval()

    # Loading train and test datasets
    train_ds = pd.read_csv(r"Quora_Is_Duplicate_Detection\datasets\train_ds.csv")
    test_ds = pd.read_csv(r"Quora_Is_Duplicate_Detection\datasets\test_ds.csv")

    # Creating datasets and dataloaders
    train_dataset = QuoraDataset(config, train_ds, config.maxlen)
    test_dataset = QuoraDataset(config, test_ds, config.maxlen)
    traindataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valdataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # Example usage of the functions/classes:
    class_names = ["Not duplicate", "duplicate"]
    
    # Plot ROC curve and confusion matrix for train and test sets
    print("On Training dataset")
    cm_train = print_report(traindataloader, model, plot_filename="Quora_Is_Duplicate_Detection/plots/roc_train.png")
    plot_confusion_matrix(cm_train, class_names, 'Quora_Is_Duplicate_Detection/plots/cm_train.png')
    
    print("On Test Dataset")
    cm_test = print_report(valdataloader, model, plot_filename="Quora_Is_Duplicate_Detection/plots/roc_test.png")
    plot_confusion_matrix(cm_test, class_names, 'Quora_Is_Duplicate_Detection/plots/cm_test.png')
    
    model_path = "Quora_Is_Duplicate_Detection/quoraduplidetec4m.pth"
    threshold = 0.7
    # Get accuracy and F1 score with a specific threshold
    accuracy, f1score, confusion_matrix = get_accuracy_with_threshold(config, test_ds, model_path, device, threshold=threshold, sample_size=5000)
    plot_confusion_matrix(confusion_matrix, class_names, f"Quora_Is_Duplicate_Detection/plots/cm_with_threhold{threshold}.png")
    
    # Predict specific questions
    questions1 = test_ds['question1'].values[50:60]
    questions2 = test_ds['question2'].values[50:60]
    real = test_ds['is_duplicate'].values[50:60]
    pipeline = PipeLine(config, model_path=model_path)
    
    results = pipeline.predict(questions1, questions2, real=real, threshold=0.7, verbose=True)