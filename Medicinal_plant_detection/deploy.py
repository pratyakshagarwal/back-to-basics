import streamlit as st
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

from config import ModelConfig
from data import idx_to_class
from model import CNN_Model

def visualize_image(image, class_name):
    """
    Function to visualize the image along with its predicted class name.
    The image is denormalized for proper visualization.
    
    Parameters:
    image (Tensor): The input image tensor.
    class_name (str): The predicted class name for the image.
    """
    # Undo normalization for displaying the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image.numpy().transpose((1, 2, 0))  # Convert from CHW to HWC format
    img = std * img + mean  # Denormalize
    img = np.clip(img, 0, 1)  # Clip values to ensure they are within [0, 1]

    plt.imshow(img)
    plt.title(f'Predicted: {class_name}')
    st.pyplot(plt)

# Transformation for validation images
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])

def main(model):
    """
    Main function to run the Streamlit application.
    
    Parameters:
    model (torch.nn.Module): The trained PyTorch model for image classification.
    """
    st.title("Image Classification with Streamlit")
    
    # Upload an image file
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("")
        st.write("Classifying...")
        
        # Preprocess the image
        input_image = val_transform(image)
        input_image = input_image.unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_image)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            class_name = idx_to_class[predicted.item()]  # Convert index to class name
        
        st.write(f'Predicted class: {class_name}')
        visualize_image(input_image.squeeze(), class_name)  # Visualize the image with the predicted class

if __name__ == '__main__':
    # Check for GPU availability and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model configuration and initialize the model
    config = ModelConfig()
    model = CNN_Model(config)
    
    # Load the trained model weights
    model = model.load_model(filepath="models/medplantdetec_6m.pth", config=config).to(device)
    model.eval()  # Set the model to evaluation mode
    
    # Run the Streamlit application
    main(model)