# cifar10_classification
This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 categories. The model is built using PyTorch and achieves a test accuracy of 72.72%. This repository contains the Python code, model weights, performance metrics, and visualizations. Designed for educational and benchmarking purposes.
# CIFAR-10 Classification Using CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 categories. The CIFAR-10 dataset is widely used for benchmarking machine learning models. The model was trained using PyTorch and achieves an accuracy of **72.72%** on the test set.

---

## Dataset
The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60,000 32x32 color images in 10 classes:
- **Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
- **Train/Test Split:** 50,000 training images and 10,000 test images.

---

## Model Architecture
The implemented CNN consists of:
1. **Convolutional Layers:** Two convolutional layers with ReLU activation.
2. **Pooling Layers:** Max-pooling after each convolutional layer.
3. **Fully Connected Layers:** Two fully connected layers with a dropout of 0.5.
4. **Output Layer:** A softmax activation layer for classification into 10 categories.

---

## Results
- **Test Accuracy:** 72.72%
- **Class-wise Accuracy:** Detailed in `class_accuracy.csv`.
- **Confusion Matrix:** Available for visualizing the model's performance across all classes.

---

## Visualizations
1. **Training Loss Plot:**
   ![Training Loss](path-to-loss-plot.png)
2. **Confusion Matrix:**
   ![Confusion Matrix](path-to-confusion-matrix.png)

---

## Files in the Repository
1. **`cifar10_cnn.py`**: Python script for training and evaluating the CNN.
2. **`cifar10_project.ipynb`**: Jupyter Notebook with the complete code for training, evaluation, and visualization.
3. **`cifar10_cnn.pth`**: Saved model weights.
4. **`accuracy.txt`**: File containing the overall test accuracy.
5. **`class_accuracy.csv`**: Class-wise accuracy results.
6. **`summary.txt`**: Summary of the model's performance.
7. **`README.md`**: This documentation file.

---

## How to Run

### **Clone the Repository**
```bash
git clone https://github.com/SajadWaloo/cifar10_classification.git
cd cifar10_classification
