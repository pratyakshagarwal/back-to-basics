import torch
from torchvision import transforms, datasets
from torch.utils.data import random_split
import matplotlib.pyplot as plt

# Define transformations for the training and validation sets
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224
    transforms.RandomRotation(degrees=25),  # Random rotation up to 25 degrees
    transforms.ColorJitter(brightness=0.25),  # Adjust brightness with a factor of 0.25
    transforms.RandomHorizontalFlip(p=0.25),  # Random horizontal flip with a probability of 0.25
    transforms.RandomVerticalFlip(p=0.05),  # Random vertical flip with a probability of 0.05
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.RandomGrayscale(p=0.01),  # Randomly convert the image to grayscale with a probability of 0.01
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])

def train_test_split(dataset, train_ratio=0.8):
    """
    Splits the dataset into training and test sets.

    Args:
    - dataset (torch.utils.data.Dataset): The dataset to be split.
    - train_ratio (float): Ratio of the dataset to be allocated for training (default: 0.8).

    Returns:
    - train_dataset (torch.utils.data.Subset): Subset of the dataset for training.
    - test_dataset (torch.utils.data.Subset): Subset of the dataset for testing.
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


# Load the dataset with the training transformations
train_fld_pth = r"datasets\train"
val_fld_pth = r"datasets\valid"
test_fld_pth = r"datasets\test"

train_dataset = datasets.ImageFolder(root=train_fld_pth, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_fld_pth, transform=val_transform)
test_dataset = datasets.ImageFolder(root=test_fld_pth, transform=val_transform)

# Invert the class_to_idx dictionary to get idx_to_class
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}


def visualize_image(dataset, idx_to_class):
    """
    Visualizes a random image from the dataset.

    Args:
    - dataset (torch.utils.data.Dataset): The dataset containing the images.
    - idx_to_class (dict): Dictionary mapping class indices to class names.

    Returns:
    None
    """
    idx = torch.randint(len(dataset), size=(1,))
    image, label = dataset[idx.item()]

    # Undo normalization for displaying the image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image * std + mean  # Reverse the normalization
    img = img.permute(1, 2, 0)  # Change the shape to (224, 224, 3) for displaying

    class_name = idx_to_class[label]

    # Display the image
    plt.imshow(img)
    plt.title(f'Class: {class_name}')
    plt.show()

    # Print label and class name
    print(f'Label: {label}, Class name: {class_name}')


if __name__ == '__main__':
    # Print dataset sizes to verify
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Testing dataset size: {len(test_dataset)}')

    # Print number of classes
    print(f'Number of classes:', len(idx_to_class))

    # Verify the transformations
    print("Train Transformations:", train_dataset.transform)
    print("Validation Transformations:", val_dataset.transform)
    print("Testing Transformations:", test_dataset.transform)

    # Visualize a sample image from train_dataset and test_dataset
    visualize_image(train_dataset, idx_to_class)
    visualize_image(test_dataset, idx_to_class)