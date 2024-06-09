import torch
import torch.nn as nn

from config import ModelConfig  # Importing the ModelConfig class from the config module

class CNN_Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Define the CNN layers
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=config.in_channels,
                      out_channels=config.out_channels,
                      kernel_size=config.kernel_size,
                      stride=config.stride,
                      padding=config.padding),
            nn.BatchNorm2d(num_features=config.out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=config.pool_kernel_size, stride=config.pool_stride),

            nn.Conv2d(in_channels=config.out_channels,
                      out_channels=config.out_channels*2,
                      kernel_size=config.kernel_size,
                      stride=config.stride,
                      padding=config.padding),
            nn.BatchNorm2d(num_features=config.out_channels*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=config.pool_kernel_size, stride=config.pool_stride),

            nn.Flatten()  # Flatten the output for the fully connected layers
        )

        # Compute the size of the input to the first fully connected layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, config.in_channels, config.img_size, config.img_size)  # Assuming input images are 224x224
            dummy_output = self.cnn_model(dummy_input)
            n_features = dummy_output.shape[1]

        # Define the fully connected layers
        self.fc_model = nn.Sequential(
            nn.Linear(n_features, config.hidden_dim),
            nn.BatchNorm1d(num_features=config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes),
            nn.LogSoftmax(dim=1)  # LogSoftmax for numerical stability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the model
        x = self.cnn_model(x)
        x = self.fc_model(x)
        return x
    
    def save_model(self, filepath):
        # Save the model's state dictionary to a file
        torch.save(self.state_dict(), filepath)

    @classmethod
    def load_model(cls, filepath, config, map_location=torch.device('cpu')):
        # Load a model from a file
        model = cls(config)  # Create an instance of the model
        model.load_state_dict(torch.load(filepath, map_location=map_location))  # Load the model's state dictionary
        model.eval()  # Set the model to evaluation mode
        return model
    
if __name__ == '__main__':
    pass  # This block is executed only if the script is run directly, not when imported as a module
