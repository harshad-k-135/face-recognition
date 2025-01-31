# Face Recognition Model Using PyTorch

## Overview
This project implements a deep learning-based **Face Recognition Model** using **PyTorch**. The model is designed to classify images of individuals into one of **31 distinct classes**. The dataset consists of images stored in a hierarchical directory structure within Kaggle - [Dataset](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset). The model achieves an exceptional accuracy of **99.68%**, making it highly reliable for real-world applications.

The training process involves **data preprocessing, augmentation, model architecture design, training, evaluation, and visualization** of results. The model is optimized for **GPU/TPU acceleration** to handle large-scale face datasets efficiently.

---

## Dataset Structure
The dataset is stored in my local stoage and follows the structure:
```
/DATASETS/Faces/
    ├── Person_Name_1/
    │   ├── person_name_1_001.jpg
    │   ├── person_name_1_002.jpg
    │   ├── ...
    ├── Person_Name_2/
    │   ├── person_name_2_001.jpg
    │   ├── person_name_2_002.jpg
    │   ├── ...
    ├── ...
```
Each subdirectory corresponds to a unique person, and the images inside contain different facial variations of that individual.

---

## Model Architecture
The model is built using PyTorch and leverages a pretrained **ResNet50** as its backbone. The architecture is designed for high accuracy and efficient learning, incorporating fine-tuning, regularization, and advanced optimization techniques. 

### Key Features of the Model: 
- **Resnet50 backbone**: Extract spatial features from the input images
- **Fine tuning strategy**: The last 3 layers of the model are unfrozen, allowing the model to adapt to the new dataset, while using the pretrained weights.
- **FC layer**: The original classification head is replaced with a custom **two-layered FC network**.
- **Weight Initialization**: Xavier Uniform Initialization is applied to fully connected layers for stable learning. 
- **Loss Function**: Uses Cross-Entropy Loss with Label Smoothing (0.1) to improve generalization and mitigate overconfidence.
- **Optimizer**: Utilizes AdamW with learning rate = 0.0003 and weight decay = 1e-4 for efficient gradient updates.
- **Learning rate scheduler**:  Implements Cosine Annealing LR scheduler with T_max=10 and eta_min=1e-6, enabling dynamic learning rate adjustments for improved convergence.

---

## Training & Evaluation
The model was trained on a local machine using **GPU acceleration (CUDA)**. The dataset was automatically split into **training and validation sets** using **80% training and 20% validation**. The model was trained for **12 epochs** with a batch size of **32**.

### **Performance Metrics:**
- **Final Accuracy**: **99.68%**
- **Loss**: Minimized using AdamW optimization
- **Correct Predictions in Test Set (20 random samples)**: **19/20 correctly classified**

The training process included **data augmentation techniques** such as:
- **Random rotation**
- **Horizontal flipping**
- **Brightness and contrast adjustments**

---

## Results & Visualizations
### **1. Accuracy vs. Epochs Graph**
![Accuracy-epoch](https://github.com/harshad-k-135/face-recognition/blob/main/output.png?raw=true)

The graph above demonstrates the steady increase in accuracy as the model progresses through training epochs, achieving near-perfect classification at the final stage.

### **2. True vs. Predicted Labels (20 Random Samples)**
![True vs Predicted Labels]([image-1.png](https://github.com/harshad-k-135/face-recognition/blob/main/pred.png?raw=true)

This image visualizes 20 randomly selected test images from the dataset. **19 out of 20 predictions were correct**, demonstrating the model’s high reliability.

---

## License
This project is licensed under the **MIT License**, allowing open-source use and modifications while requiring attribution.

---
