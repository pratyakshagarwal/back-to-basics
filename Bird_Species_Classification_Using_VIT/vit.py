import warnings
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

class Patch_Embedding(nn.Module):
    """
    A module that converts an image into patch embeddings.

    Args:
        in_channels (int): Number of input channels in the image. Default is 3 for RGB images.
        patch_size (int): Size of each patch. Default is 16.
        embedding_dim (int): Dimension of the embedding for each patch. Default is 768.
    """

    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)  # Convolutional layer to create patches

        self.flatten = nn.Flatten(start_dim=2, end_dim=3)  # Flatten patches into a single dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Patch_Embedding module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, embedding_dim).
        """
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input layer must be divisible by {self.patch_size}"

        # Ensure input is of type float
        x = x.float()

        x_patched = self.patcher(x)  # Apply the convolution to create patches
        x_flattened = self.flatten(x_patched)  # Flatten the patches from (N, C, H, W) to (N, C, H*W)
        return x_flattened.permute(0, 2, 1)  # Rearrange dimensions to (batch_size, num_patches, embedding_dim)

class VIT(nn.Module):
    """
    Vision Transformer (ViT) model.

    Args:
        img_size (int): Size of the input image (assumes square images). Default is 224.
        num_channels (int): Number of input channels in the image. Default is 3 for RGB images.
        patch_size (int): Size of each patch. Default is 16.
        embedding_dim (int): Dimension of the embedding for each patch. Default is 768.
        dropout (float): Dropout rate. Default is 0.1.
        mlp_size (int): Size of the MLP layer in the transformer. Default is 3072.
        nlayer (int): Number of transformer encoder layers. Default is 12.
        nheads (int): Number of attention heads. Default is 12.
        num_classes (int): Number of output classes. Default is 1000.
    """

    def __init__(self,
                 config,
                 ) -> None:
        super().__init__()

        assert config.img_size % config.patch_size == 0, "The image size must be divisible by patch size"

        # Initialize patch embedding layer
        self.patch_embedding_layer = Patch_Embedding(in_channels=config.num_channels,
                                                     patch_size=config.patch_size,
                                                     embedding_dim=config.embedding_dim)

        # Class token parameter
        self.class_token = nn.Parameter(torch.randn(1, 1, config.embedding_dim), requires_grad=True)

        # Calculate the number of patches
        num_patches = (config.img_size * config.img_size) // (config.patch_size ** 2)

        # Positional embeddings parameter
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.embedding_dim), requires_grad=True)

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.nheads,
                dim_feedforward=config.mlp_size,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ),
            num_layers=config.nlayer
        )

        self.dropout = nn.Dropout(config.dropout)

        # MLP head for classification
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, img_size, img_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        batch_size = x.shape[0]

        # Obtain patch embeddings
        patch_embd = self.patch_embedding_layer(x)

        # Expand class token to batch size and concatenate with patch embeddings
        class_token = self.class_token.expand(batch_size, -1, -1)
        h = torch.cat((class_token, patch_embd), dim=1)

        # Add positional embeddings to the concatenated tensor
        pos_embd = self.positional_embeddings + h

        # Pass through transformer encoder
        encoder_out = self.transformer_encoder(self.dropout(pos_embd))

        # Pass the output of the [CLS] token through the MLP layer
        out = self.mlp_layer(encoder_out[:, 0])

        return out

    def save_model(self, filename) -> None:
        """
        Save the model state

        Parameter: 
        filename (str) : the name by which the model is going to be saved
        """
        # Save the state dictionary
        torch.save(self.state_dict(), filename)

    @classmethod
    def load_model(cls, config,  filepath):
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
    # Sample usage
    sample_model = VIT(num_classes=3)
    sample_input = torch.randn(size=(1, 3, 224, 224))  # Note: batch size is 1
    sample_output = sample_model(sample_input)
    print(f"input shape: {sample_input.shape}")
    print(f"output shape: {sample_output.shape}")
    # print(f"output: {sample_output}")
