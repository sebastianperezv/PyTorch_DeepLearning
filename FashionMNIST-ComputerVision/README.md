# PyTorch Computer Vision: FashionMNIST Classification 👗✨

## Project Description
This repository contains a Jupyter Notebook detailing a Computer Vision project implemented with **PyTorch**. The primary goal is to build, train, and evaluate neural network models, focusing on a **Convolutional Neural Network (CNN) 🧠**, to classify images from the renowned **FashionMNIST** dataset.

The FashionMNIST dataset consists of 70,000 grayscale images (28x28 pixels) of 10 different types of clothing items (e.g., T-shirt, Trouser, Sneaker).

---

## Model Architectures 🏗️
The notebook explores a progression of models for the image classification task:
1.  **Baseline Model (Model V0):** A simple **Linear Model** (Multi-Layer Perceptron) used as an initial reference point.
2.  **Convolutional Neural Network (CNN) (Model V2):** An in-depth implementation of a CNN. This architecture utilizes sequential **convolutional layers**, **ReLU** activation, and **Max Pooling** to effectively capture spatial and hierarchical features from the images.

---

## Key Concepts Covered 📚
This project is a practical guide that demonstrates fundamental concepts in deep learning and PyTorch:
* **Data Preparation:** Using `torchvision.datasets` and `DataLoader` for efficient data loading and batching. 📦
* **Model Building:** Defining custom models using PyTorch's `nn.Module` and `nn.Sequential`.
* **Training Loop Implementation:** Implementing the complete forward pass, loss calculation, **backward pass** (`loss.backward()`), and optimization cycle (`optimizer.step()`). ⚙️
* **Model Evaluation:** Using `torch.inference_mode()` for efficient testing and calculating metrics like **Loss** and **Accuracy**. 🎯
* **Model Persistence:** Saving and loading trained model state dictionaries for future use. 💾

---

## Performance Snapshot 📈
The trained **Convolutional Neural Network (Model V2)** achieved robust performance on the unseen test dataset:

| Model Name | Test Loss | Test Accuracy |
| :--- | :--- | :--- |
| **FashionMNISTModelV2 (CNN)** | ~0.276 | **~90.38%** |
