1. # **Credit Card Fraud Detection**

This repository contains code and resources for a credit card fraud detection model. The model uses a neural network to identify fraudulent transactions from a dataset of credit card transactions

### **Table of Contents**

- Installation
- Usage
- Model Architecture
- Data Preprocessing
- Training the Model
- Evaluation
- Results
- Important Notes

### **Installation**
To run this project, you need to have Python and the necessary libraries installed. You can install the required libraries using the following command:
```bash
pip install -r requirements.txt
```

### **Usage**
1. clone the repositry
```bash
git clone https://github.com/your-username/CreditCard_Fraud_Detection.git
cd CreditCard_Fraud_Detection
```

2. Place the `creditcardfraud.csv` dataset in the datasets directory.

3. Run the training script:
```bash
python train.py
```

4. Run the evaluation script:
```bash
python evaluate.py
```

### **Model Architecture**
The model is a multi-layer perceptron (MLP) neural network with the following layers:

- Input layer with 29 features
- Three hidden layers with ReLU activation and batch normalization
- Dropout layers for regularization
- Output layer with sigmoid activation

### **Data Preprocessing**
- The dataset is split into training and testing sets with an 80-20 ratio.
- The dataset is balanced using the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to handle class imbalance.
- Features are extracted and labels are separated for model training.

### **Training the Model**
The model is trained using the Binary Cross-Entropy (BCE) loss function with class weights to handle class imbalance. The optimizer used is Adam with a learning rate of 0.01. The training script `train.py` handles the training process, and the model is saved to `fraudetcmodel181k.pth`.

#### **Training Script**
- train.py: Handles data loading, model training, and saving the trained model.

### **Evaluation**
The model is evaluated on the test set, and metrics such as accuracy and F1 score are calculated. A confusion matrix is also plotted to visualize the performance.

### **Evaluation Script**
- `evaluate.py`: Loads the test data and trained model, and prints the classification report and confusion matrix.

### **Results**
- The model achieved an accuracy of 0.9070 and an F1 score of 0.8959 on a balanced test set.
- The confusion matrix and classification report provide a detailed breakdown of the model's performance.

### **Important Notes**
- **Threshold Importance**: The threshold for predicting fraud is set to 0.3. This threshold was chosen to balance precision and recall for the minority class (fraudulent transactions). Adjusting this threshold can significantly impact the model's performance, especially in handling imbalanced datasets.

### **Contributing**
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

### **License**
This project is licensed under the MIT License.


2. # **Medicinal Plant Classification**:

  This project involves building a Convolutional Neural Network (CNN) to 
classify medicinal plants using images. The model is implemented using PyTorch and includes data preprocessing, model training, and evaluation components. Additionally, a Streamlit application is provided for deploying the model.

### **Table of Contents**

- Installation
- Dataset
- Configuration
- Training the Model
- Evaluating the Model
- Deploying the Model
- Model Key Points
- Usage
- Results
- References

### **Installation**

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medicinal-plant-classification.git
cd medicinal-plant-classification
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

### **Dataset**
The dataset used in this project is a collection of medicinal plant images. The images should be organized in the following structure:

```markdown
Medicinal plant dataset/
    class1/
        img1.jpg
        img2.jpg
        ...
    class2/
        img1.jpg
        img2.jpg
        ...
    ...
```

### **Configuration**
The model configuration is specified in `config.py` using a dataclass `ModelConfig`. This includes hyperparameters like image size, kernel size, number of channels, and learning rate.

```python 
from dataclasses import dataclass
from data import idx_to_class

@dataclass
class ModelConfig():
    img_size: int = 224
    kernel_size: int = 3
    in_channels: int = 3
    out_channels: int = 8
    pool_kernel_size: int = 2
    pool_stride: int = 2
    num_classes: int = len(idx_to_class)
    hidden_dim: int = 128
    stride: int = 1
    padding: int = 1
    dropout: float = 0.25
    batch_size: int = 64
    lr: float = 1e-3
```

### Training the Model
To train the model, run `train.py`. This script initializes the model, dataloaders, and the training loop.

```bash
python train.py
```

**Example Output:**
```yaml
using cuda
Epoch: 1  |  Train Loss: 3.1795 | Val Loss: 2.6618  |  Train Accuracy: 0.1791  |  Val Accuracy: 0.3078
...
```

### **Evaluating the Model**
To evaluate the model and visualize predictions, run `evaluate.py`. This script loads the saved model and displays a batch of predictions.

```bash
python evaluate.py
```

### **Deploying the Model**
A Streamlit application is provided for deploying the model. To run the app, use the following command:

```bash
streamlit run deploy.py
```
Upload an image through the web interface, and the model will predict the class of the medicinal plant in the image.

### **Model Key Points**

- Architecture: The model is a CNN consisting of two convolutional layers, each followed by batch normalization, ReLU activation, and max pooling. This is followed by a fully connected network.
- Input Size: The input images are resized to 224x224 pixels.
- Output: The final layer is a softmax layer that outputs the probability distribution over the classes.
- Optimization: The model is optimized using the Adam optimizer with a learning rate scheduler.
- Loss Function: Negative Log-Likelihood Loss (NLLLoss) is used for training.
- Metrics: Accuracy is measured using TorchMetrics.

### **Usage**
- Load and Use the Model in Your Code:

```python 
import torch
from config import ModelConfig
from model import CNN_Model

# Initialize the configuration and model
config = ModelConfig()
model = CNN_Model(config)

# Load the trained model weights
model.load_model(filepath="models/medplantdetec_6m.pth", config=config)
```

### **Results**
The model was trained for 24 epochs with the following results:
```yaml
Epoch: 40  |  Train Loss: 0.1841 | Val Loss: 0.1050  |  Train Accuracy: 0.9603  |  Val Accuracy: 0.9821
```

### **References**
- PyTorch Documentation
- Streamlit Documentation
- Torchvision Transforms