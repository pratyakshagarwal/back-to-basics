import torch
import torch.nn as nn

class FraudModel(nn.Module):
    """
    A neural network model for fraud detection.

    This model is built using a Multi-Layer Perceptron (MLP) architecture with 
    several hidden layers. Each hidden layer consists of a Linear layer, 
    ReLU activation, Batch Normalization, and Dropout for regularization.
    
    Parameters:
    config (object): A configuration object containing model hyperparameters:
        - in_features (int): Number of input features.
        - hidden_dim (int): Dimension of hidden layers.
        - dropout (float): Dropout rate for regularization.
        - out_features (int): Number of output features (typically 1 for binary classification).
    """

    def __init__(self, config):
        """
        Initializes the FraudModel with the given configuration.

        Parameters:
        config (object): Configuration object with model hyperparameters.
        """
        super().__init__()

        # Define the MLP architecture
        self.mlp = nn.Sequential(
            nn.Linear(config.in_features, config.hidden_dim * 4),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim * 4),
            nn.Dropout(config.dropout),

            nn.Linear(config.hidden_dim * 4, config.hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim * 2),
            nn.Dropout(config.dropout),

            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Dropout(config.dropout),

            nn.Linear(config.hidden_dim, config.out_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the MLP.
        """
        return self.mlp(x)

    def save_model(self, filepath: str) -> None:
        """
        Saves the model state to a file.

        Parameters:
        filepath (str): The path where the model state will be saved.
        """
        torch.save(self.state_dict(), filepath)

    @classmethod
    def load_model(cls, filepath: str, config) -> 'FraudModel':
        """
        Loads the model state from a file.

        Parameters:
        filepath (str): The path from where the model state will be loaded.
        config (object): Configuration object to initialize the model.

        Returns:
        FraudModel: The model loaded with the saved state.
        """
        model = cls(config)
        model.load_state_dict(torch.load(filepath))
        model.eval()  # Set the model to evaluation mode
        return model
