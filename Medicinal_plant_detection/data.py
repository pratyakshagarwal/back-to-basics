from torchvision import datasets, transforms
from torch.utils.data import random_split

# Define transformations for the training and validation sets
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224
    transforms.RandomRotation(degrees=25),  # Apply random rotation to the images
    transforms.ColorJitter(brightness=0.25),  # Adjust brightness randomly
    transforms.RandomHorizontalFlip(p=0.25),  # Apply random horizontal flip with probability 0.25
    transforms.RandomVerticalFlip(p=0.05),  # Apply random vertical flip with probability 0.05
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.RandomGrayscale(p=0.01),  # Apply random grayscale transformation with probability 0.01
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])

def train_test_split(dataset, train_ratio=0.8):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    
    # Split the dataset into training and validation sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset

# Load the dataset with the training transformations
train_fld_pth = "Medicinal_plant_detection\Medicinal_plant_dataset"
dataset = datasets.ImageFolder(root=train_fld_pth, transform=train_transform)

# Split the dataset into training and validation sets
train_dataset, _ = train_test_split(dataset)

# Load the dataset with the validation transformations
dataset = datasets.ImageFolder(root=train_fld_pth, transform=val_transform)
_, test_dataset = train_test_split(dataset)

# Invert the class_to_idx dictionary to get idx_to_class
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

if __name__ == '__main__':
    # Print dataset sizes to verify
    print(f'Total dataset size: {len(dataset)}')
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(test_dataset)}')
    
    # Verify the transformations
    print("Train Transformations:", train_dataset.dataset.transform)
    print("Validation Transformations:", test_dataset.dataset.transform)
