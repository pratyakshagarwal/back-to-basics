

# 1. **Credit Card Fraud Detection**

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


# 2. **Medicinal Plant Classification**:

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


# 3. **Quora Duplicate Question Detection**
This project focuses on detecting duplicate questions in the Quora dataset using an LSTM model. The pipeline includes preprocessing, model training, evaluation, and visualization.

### **Project Structure**

```arduino
Quora_Is_Duplicate_Detection/
│
├── datasets/
│   ├── quora_questions.csv
│   ├── train_ds.csv
│   └── test_ds.csv
│
├── plots/
│   ├── loss_plot.png
│   ├── accuracy_plot.png
│   ├── roc_train.png
│   ├── cm_train.png
│   ├── roc_test.png
│   ├── cm_test.png
│   └── cm_with_threshold{threshold}.png
│
├── config.py
├── data.py
├── evaluate.py
├── model.py
├── train.py
└── utils.py
```

### **Installation**
1. Clone the repository:
```bash
git clone https://github.com/your-username/Quora_Is_Duplicate_Detection.git
cd Quora_Is_Duplicate_Detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### **Configuration**
The configuration settings for the model are specified in the `config.py` file. Modify the `ModelConfig` class to set parameters such as learning rate, batch size, number of epochs, etc.

### **Data Preparation**
Place the `quora_questions.csv` dataset in the `datasets` directory. The script will split this dataset into training and testing datasets.

### **Training**
To train the model, run the train.py script:

This script will:

- Load and preprocess the data.
- Train the LSTM model.
- Save the trained model and training history.
- Plot training and validation loss and accuracy.

### **Evaluation**
To evaluate the model, run the `evaluate.py` script:
```bash
python evaluate.py
```
This script will:

- Load the trained model.
- Evaluate the model on the training and testing datasets.
- Print the classification report.
- Plot ROC curves and confusion matrices.

### **Utils**
The utils.py script contains utility functions for:

- Tokenizer creation and loading.
- Weighted binary cross-entropy loss calculation.
- Plotting training history, ROC curves, and confusion matrices.
- Model prediction pipeline.

### *Example Usage
Predicting Specific Questions
Use the **PipeLine** class to predict if pairs of questions are duplicates:

```python
from utils import PipeLine
from config import ModelConfig

config = ModelConfig()
model_path = "Quora_Is_Duplicate_Detection/quoraduplidetec4m.pth"

pipeline = PipeLine(config, model_path=model_path)

questions1 = ["What is the best way to learn Python?", "How do I start learning Python?"]
questions2 = ["How to learn Python programming?", "What is the best way to learn programming?"]

results = pipeline.predict(questions1, questions2, threshold=0.7)
print(results)
```

### **Getting Accuracy and F1 Score with Specific Threshold**
```python
from utils import get_accuracy_with_threshold
from config import ModelConfig

config = ModelConfig()
ds = pd.read_csv("Quora_Is_Duplicate_Detection/datasets/test_ds.csv")
model_path = "Quora_Is_Duplicate_Detection/quoraduplidetec4m.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = 0.7

accuracy, f1score, confusion_matrix = get_accuracy_with_threshold(config, ds, model_path, device, threshold=threshold, sample_size=5000)
```

### **Using the Pipeline Class for Predictions**
The PipeLine class in `utils.py` provides a convenient way to preprocess input questions, make predictions using the trained model, and display the results.

**Example usage**
```python 
from config import ModelConfig
from utils import PipeLine

config = ModelConfig()
model_path = "path_to_trained_model.pth"
pipeline = PipeLine(config, model_path=model_path)

questions1 = ["What is the capital of France?", "How to learn Python?"]
questions2 = ["Which city is the capital of France?", "How can I start learning Python?"]

results = pipeline.predict(questions1, questions2, threshold=0.5, verbose=True)
```

### **License**
This project is licensed under the MIT License. See the LICENSE file for more details.

### Acknowledgements
- Quora for providing the dataset.
- Hugging Face Tokenizers for the tokenizer library.
- PyTorch for the deep learning framework.

##### **Feel free to contribute to this project by creating issues or pull requests. Happy coding!**



# **4. Vision Transformers for Bird Species Classification**
>  **Table of Contents**
- Introduction
- Dataset
- Model Architecture
- Repo Overview
- Training
- Setup
- TensorBoard Visualization
- Results
- References

### **Introduction**
>  This project focuses on classifying bird species using **Vision Transformers** (ViTs). It leverages a dataset of **100 bird species** available on Kaggle and demonstrates the entire pipeline from training to deployment.

### **Dataset**
>  The dataset used in this project is the "100 Bird Species" dataset from Kaggle. It includes images of 100 different bird species with labeled training and test data.

`Dataset Link` - [100-bird-species](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)

### **Model Architecture**
>   The model is built using Vision Transformers (ViT), which is a state-of-the-art architecture for image classification tasks.

- **Patch Embedding**: Converts input images into patch embeddings using a convolutional layer.
- **Class Token**: Adds a learnable class token to the sequence of patch embeddings.
- **Positional Embeddings**: Adds learnable positional embeddings to the patch embeddings.
- **Transformer Encoder**: Processes the patch embeddings using multiple Transformer encoder layers.
- **MLP Head**: The output of the [CLS] token is passed through an MLP for classification.

### **Repo Overview**
> The project structure is organized as follows:

- **data**/: Directory containing dataset loading scripts (`train_dataset.py`, `val_dataset.py`, `test_dataset.py`) and utilities for data preprocessing.
- models/: Contains the ViT model architecture (`vit.py`).
- utils.py: Utility functions for plotting training history (`plot_history`), cosine learning rate scheduling (`CosineLrScheduler`), and visualization of model predictions (`visualize_predictions`).
- vit_trainer.py: Defines`VIT_Trainer` class responsible for model training (`run` method) and validation loops (`validation_loop` method).
- config.py: Configuration file (`VITConfig` dataclass) containing model hyperparameters, training configurations, and paths for logging and checkpoints.
- train.py: Script to initiate and run the training process. It imports datasets, initializes the trainer, compiles the model, and executes training for a specified number of epochs.
- evaluate.py: Script to load a trained model and visualize its predictions on training, validation, and test datasets.

### **Training**
>  The training of the Vision Transformer (ViT) model was conducted with a focus on efficient learning and robust evaluation. Below is an overview of the training setup and results:

##### **Training Configuration**
-  Model Architecture: Vision Transformer (ViT)
-  Dataset: 100 bird species dataset from Kaggle
-  Device: CUDA (if available) or CPU
-  Optimizer: AdamW
-  Learning Rate Scheduler: Custom Cosine Learning Rate Scheduler with Warmup
-  Batch Size: 64
-  Number of Epochs: 25

### **Setup**
>  Clone the repository:
```bash
git clone https://github.com/pratyakshagarwal/back-to-basics
cd Bird_Species_Classification_Using_VIT
```

>  Install dependencies:
```bash
pip install -r requirements.txt
```

### **TensorBoard Visualization**
-  To visualize training progress and metrics using TensorBoard, run:
```bash
tensorboard --logdir=runs
```
- Open your browser and go to `localhost:6006` to view the TensorBoard dashboard.

### **Results**
- The ViT model achieves competitive accuracy on the validation dataset, reaching an accuracy of 86.12% after 22 epochs of training.

### **References**
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html): Official documentation for PyTorch framework.
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929) : Original research paper introducing Vision Transformers.

