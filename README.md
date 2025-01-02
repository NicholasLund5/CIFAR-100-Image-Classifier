
# CIFAR-100 Image Classifier

This repository contains an implementation of a CIFAR-100 image classification system, featuring a ResNet-inspired architecture with Squeeze-and-Excitation (SE) blocks. This project includes both the training pipeline and a deployment-ready FastAPI service for image classification.

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Training Methodology](#training-methodology)
4. [FastAPI Backend](#fastapi-backend)
5. [Frontend Application](#frontend-application)
6. [Setup and Usage](#setup-and-usage)
7. [References](#references)

---

## Overview

The CIFAR-100 dataset consists of 100 classes with 600 images per class. This project employs a custom neural network model with residual and squeeze-and-excitation blocks to achieve high classification accuracy. The frontend provides drag-and-drop functionality for image upload, while the backend is implemented using FastAPI.

---

## Model Architecture

### Components:
1. **ResidualSEBlock**: Combines residual connections with Squeeze-and-Excitation for adaptive feature recalibration.
2. **Squeeze-Excitation Block**: Implements channel-wise attention by using global average pooling, followed by a two-layer fully connected network.
3. **Stem**: Initial convolutional layer to process input images.
4. **Stages**: Three stages with increasing number of channels and resolution reduction.
5. **Classifier**: Fully connected layer to produce predictions for 100 classes.

### Model Code:
```python
class CIFAR100Model(nn.Module):
    def __init__(self, input_shape: int, width_multiplier: int, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.stage1 = self._make_stage(16, 16 * width_multiplier, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(16 * width_multiplier, 32 * width_multiplier, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(32 * width_multiplier, 64 * width_multiplier, num_blocks=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(64 * width_multiplier, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            layers.append(
                ResidualSEBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    stride=stride if i == 0 else 1
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
```

---

## Training Methodology

### Dataset:
- CIFAR-100, consisting of 32x32 images in 100 classes.

### Preprocessing:
- Resize to 32x32 (if not already).
- Normalize with CIFAR-100-specific means and standard deviations:
  - Mean: `(0.5071, 0.4865, 0.4409)`
  - Std: `(0.2673, 0.2564, 0.2762)`

### Data Augmentation:
- Random cropping
- Random horizontal flipping

### Loss Function:
- Cross-Entropy Loss

### Optimizer:
- SGD with momentum.
- Learning rate scheduling using a cosine annealing strategy.

### Hyperparameters:
- Epochs: 100
- Batch size: 128
- Learning rate: 0.1 (with decay)

### Implementation:
Training is implemented in the accompanying Jupyter Notebook. The model is saved as `CIFAR100_model.pth` after training.

---

## FastAPI Backend

The FastAPI backend serves the trained model for inference. It includes:
- An endpoint for classification: `/classify/`
- Middleware for CORS to allow interaction with the frontend.

### Backend Code:
```python
@app.post("/classify/")
async def classify_image(image: UploadFile = File(...)):
    try:
        img = Image.open(image.file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_class = torch.max(outputs, 1)
            prediction = class_names[predicted_class.item()]

        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
```

---

## Frontend Application

The frontend is a simple HTML+CSS+JavaScript application that allows users to upload or drag-and-drop an image for classification. It communicates with the FastAPI backend to obtain predictions.

### Example Interaction:
1. User drops an image into the drop zone.
2. Upon submission, the image is sent to the backend, and the predicted class is displayed.

---

## Setup and Usage

### Requirements:
1. Python 3.8+
2. PyTorch and torchvision
3. FastAPI and Uvicorn
4. Frontend-compatible browser

### Steps:
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 5000
   ```
5. Upload an image to classify.

---
