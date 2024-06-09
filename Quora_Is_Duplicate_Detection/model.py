import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, config) -> None:
        """
        Initializes the LSTM_Model object.

        Parameters:
        config (ModelConfig): Configuration object containing model and training parameters.
        """
        super().__init__()

        # Embedding layer to convert input tokens to dense vectors
        self.embedding = nn.Embedding(config.input_vocab, config.embd_dim)

        # LSTM layer for sequence modeling
        self.lstm = nn.LSTM(
            config.embd_dim, 
            config.hidden_dim, 
            num_layers=config.n_layers, 
            bidirectional=config.bidirectional, 
            dropout=config.dropout, 
            batch_first=True
        )

        # Batch normalization layer to normalize hidden states
        self.batch_norm = nn.BatchNorm1d(config.hidden_dim * 2 if config.bidirectional else config.hidden_dim)

        # ReLU activation function
        self.relu = nn.ReLU()

        # Fully connected layer for output
        self.fc = nn.Linear(config.hidden_dim * 2 if config.bidirectional else config.hidden_dim, 1)

        # Sigmoid activation function for binary classification
        self.sigmoid = nn.Sigmoid()

        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Parameters:
        x (torch.Tensor): Input tensor containing tokenized sequences.

        Returns:
        output (torch.Tensor): Output tensor containing predicted probabilities.
        """
        # Pass input through embedding layer
        embedded = self.embedding(x)

        # Pass embedded sequences through LSTM layer
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Concatenate the final forward and backward hidden states if bidirectional
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        # Apply dropout to hidden states
        hidden = self.dropout(hidden)

        # Apply batch normalization to hidden states
        hidden = self.batch_norm(hidden)

        # Apply ReLU activation function
        hidden = self.relu(hidden)

        # Pass through fully connected layer to get output
        output = self.fc(hidden)

        # Apply sigmoid activation function to get probabilities
        output = self.sigmoid(output)

        return output

    def save_model(self, filepath):
        """
        Saves the model state to a file.

        Parameters:
        filepath (str): The file path to save the model state.
        """
        torch.save(self.state_dict(), filepath)

    @classmethod
    def load_model(cls, filepath, config):
        """
        Loads the model state from a file and returns a model instance.

        Parameters:
        filepath (str): The file path to load the model state from.
        config (ModelConfig): Configuration object containing model and training parameters.

        Returns:
        model (LSTM_Model): The loaded model instance.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = cls(config)
        model.load_state_dict(torch.load(filepath, map_location=device))
        model.eval()  # Set model to evaluation mode
        return model
    
if __name__ == '__main__':
    pass
