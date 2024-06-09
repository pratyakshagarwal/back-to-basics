## **Credit Card Fraud Detection**

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