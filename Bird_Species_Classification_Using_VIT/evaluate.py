import torch

from vit import VIT_Model, VIT
from utils import visualize_predictions
from data import train_dataset, test_dataset, val_dataset, idx_to_class
from config import VITConfig

if __name__ == '__main__':
    # configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")

    # load model from the model path
    config = VITConfig()
    model_path = "vitmodel7m.pth"
    model = VIT(config)
    model = model.load_model(config, model_path)

    # make prediction on different dataset
    print(visualize_predictions(model, train_dataset, idx_to_class, device, r'plots\train_predictions.png'))
    print(visualize_predictions(model, val_dataset, idx_to_class, device, r'plots\val_predictions.png'))
    print(visualize_predictions(model, test_dataset, idx_to_class, device, r'plots\test_predictions.png'))