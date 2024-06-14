import torch
from torchmetrics.classification import Accuracy

# Assuming these imports are from your project structure
from data import train_dataset, val_dataset  # Assuming test_dataset is not used for training
from VIT_Trainer import VIT_Trainer
from config import VITConfig
from utils import plot_history

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    
    # Load configuration
    config = VITConfig()

    # Select device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize accuracy metric
    accuracy_fn = Accuracy(task="multiclass", num_classes=config.num_classes).to(device)
    
    # Set optimizer parameters
    lr = 3e-4
    epochs = 25
    optimizer_params = {'lr': lr}

    # Initialize trainer
    trainer = VIT_Trainer(config, device)
    
    # Compile trainer with optimizer and accuracy function
    trainer.compile(torch.optim.AdamW, optimizer_params=optimizer_params, accuracy_fn=accuracy_fn)
    
    # Run training
    history = trainer.run(epochs, train_dataset, val_dataset)
    
    # Plot and save the loss and accuracy graphs
    plot_history(history, 'loss', 'plots/loss_plot.png')
    plot_history(history, 'accuracy', 'plots/accuracy_plot.png')
    

#Training completed. To visualize training progress with TensorBoard, run the following command in terminal
# tensorboard --logdir runs
