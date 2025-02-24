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

## Results
After training for 100 iterations with a learning rate of 0.1, the model achieves an accuracy of around 85% on the dataset.

link to downllaod dataset : https://www.kaggle.com/competitions/digit-recognizer

