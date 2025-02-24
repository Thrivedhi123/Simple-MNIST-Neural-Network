# Simple MNIST Neural Network

## Overview
This project implements a simple neural network from scratch using NumPy to classify handwritten digits from the MNIST dataset. It does not rely on deep learning libraries like TensorFlow or PyTorch but instead builds the network step by step using matrix operations.

## Dataset
The dataset consists of 42,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. It is stored in:
- `data.csv`: Contains 42,000 labeled images with 785 columns (1 label + 784 pixel values)

## Neural Network Architecture
The neural network has two layers:
- **Input Layer:** 784 neurons (each pixel in a 28x28 image is an input feature)
- **Hidden Layer:** 10 neurons with ReLU activation
- **Output Layer:** 10 neurons with Softmax activation (one for each digit 0-9)

## Mathematical Formulation
### **1. Initialization of Parameters**
The network initializes weights and biases randomly:
- \( W_1 \in \mathbb{R}^{10 \times 784} \), \( b_1 \in \mathbb{R}^{10 \times 1} \)
- \( W_2 \in \mathbb{R}^{10 \times 10} \), \( b_2 \in \mathbb{R}^{10 \times 1} \)

### **2. Forward Propagation**
#### **Hidden Layer Computation**
1. Compute the linear transformation:
   \[ Z_1 = W_1 X + b_1 \]
2. Apply ReLU activation:
   \[ A_1 = \max(0, Z_1) \]

#### **Output Layer Computation**
1. Compute the linear transformation:
   \[ Z_2 = W_2 A_1 + b_2 \]
2. Apply Softmax activation:
   \[ A_2 = \frac{e^{Z_2}}{\sum e^{Z_2}} \]

### **3. Loss Function**
The network uses **Categorical Cross-Entropy Loss**:
\[ \mathcal{L} = - \frac{1}{m} \sum_{i=1}^{m} y_i \log(A_{2i}) \]
where:
- \( y_i \) is the one-hot encoded ground truth label
- \( A_{2i} \) is the predicted probability for the correct class
- \( m \) is the batch size

### **4. Backpropagation**
Compute gradients for updating parameters using chain rule:
1. **Output Layer Gradients**
   \[ dZ_2 = A_2 - Y \]
   \[ dW_2 = \frac{1}{m} dZ_2 A_1^T \]
   \[ db_2 = \frac{1}{m} \sum dZ_2 \]

2. **Hidden Layer Gradients**
   \[ dZ_1 = (W_2^T dZ_2) \cdot ReLU'(Z_1) \]
   \[ dW_1 = \frac{1}{m} dZ_1 X^T \]
   \[ db_1 = \frac{1}{m} \sum dZ_1 \]

### **5. Parameter Update (Gradient Descent)**
Using a learning rate \( \alpha \):
\[ W_1 = W_1 - \alpha dW_1, \quad b_1 = b_1 - \alpha db_1 \]
\[ W_2 = W_2 - \alpha dW_2, \quad b_2 = b_2 - \alpha db_2 \]

### **6. Training Loop**
The model is trained for a fixed number of iterations, updating parameters at each step. Accuracy is evaluated periodically.

### **7. Prediction and Evaluation**
After training, the network predicts labels using:
\[ \hat{y} = \arg\max(A_2) \]
Accuracy is computed as the fraction of correct predictions.

## Results
After training for 100 iterations with a learning rate of 0.1, the model achieves an accuracy of around 85% on the dataset.



