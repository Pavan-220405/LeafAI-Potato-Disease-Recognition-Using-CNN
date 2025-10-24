# 🥔 Potato Plant Disease Detection using CNN & Transfer Learning

## Overview
This project leverages **Deep Learning** to accurately identify **potato leaf diseases** — a crucial step in ensuring sustainable crop yield and food security.  
Using a **Convolutional Neural Network (CNN)** built from scratch and compared against **transfer learning (EfficientNet)**, this work demonstrates how model architecture and data augmentation significantly affect performance in real-world agricultural imaging.

---

##  Dataset
- **Source:** Kaggle – PlantVillage Dataset
- **Classes:**
  -  **Potato Early Blight** – 1000 images  
  -  **Potato Late Blight** – 1000 images  
  - **Potato Healthy** – 152 images → **augmented** to 1000 images  

After augmentation, all classes were balanced with **1000 images each (3000 total)**.

---

##  Data Preprocessing
- **Image Size:** 224 × 224 pixels  
- **Normalization:** Pixel values scaled to [0, 1] using `keras.layers.Rescaling`  
- **Augmentation Techniques:**
  - Random rotation, shear, and zoom
  - Width/height shifts
  - Brightness variation
  - Horizontal flipping
- **Data Split:**
  - 80% Training  
  - 10% Validation  
  - 10% Testing  

---

##  Model Architectures

###  Custom CNN Model
A robust CNN architecture was designed with multiple convolutional blocks for hierarchical feature extraction.

```python
model = Sequential([
    rescale,
    data_augment,

    # Block 1
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    # Block 2
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    # Block 3
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    # Fully Connected Layers
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])
```

**Training Configuration**
- Optimizer: `Adam`
- Loss Function: `Categorical Crossentropy`
- Batch Size: `32`
- Epochs: `~20`

> Adding **Batch Normalization** led to unstable convergence, so it was excluded from the final architecture.

---

###  Transfer Learning (EfficientNet)
A pre-trained **EfficientNet** model was fine-tuned for comparison.  
However, due to domain differences and small dataset size, the model **underperformed**, achieving:

| Metric | Value |
|:--|:--|
| Training Accuracy | 0.33 |
| Validation Accuracy | 0.33 |

---

## 📈 Performance Metrics

| Model | Train Acc | Val Acc | Test Acc | Test Precision | Test F1 Score |
|:------|:----------:|:--------:|:---------:|:----------:|:----------:|
| **Custom CNN** | 0.95 | 0.94 | 0.96 | 0.96 | 0.96 |
| **EfficientNet (TL)** | 0.30 | 0.30 | – | – | – |

---

##  Key Insights
-  **Custom CNN** achieved **superior accuracy (96%)**, proving that well-tuned small models can outperform complex pre-trained networks for specific domains.
-  **Batch Normalization** disrupted stability — likely due to limited batch size or aggressive learning rate.
-  **Data augmentation** played a pivotal role in balancing the dataset and improving generalization.
-  **Transfer learning** struggled due to domain mismatch (generic ImageNet features vs. leaf texture patterns).

---

##  Conclusion
- A **carefully engineered CNN** can outperform modern transfer learning models for focused, domain-specific tasks.
- Achieved **96% test accuracy** with strong precision and F1 score, validating the model’s real-world reliability.
- Demonstrates the potential of AI to **assist farmers and agronomists** in early disease detection.

---


##  Future Work  
- Experiment with **ResNet**, **InceptionV3**, and **MobileNetV2**  
- Deploy the trained model as a **web or mobile app** for real-time disease diagnosis  

---
