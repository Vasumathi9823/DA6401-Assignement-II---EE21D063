# Multi-Task Vision Pipeline: Classification, Localization & Segmentation

This repository contains a unified deep learning pipeline utilizing a **VGG11-based U-Net architecture** to perform simultaneous breed classification, object detection (bounding box regression), and semantic segmentation on the Oxford-IIIT Pet dataset.

GitHub Repo link: https://github.com/Vasumathi9823/DA6401-Assignement-II---EE21D063/tree/main

---

##  Project Overview

Our architecture leverages a shared VGG11 encoder with three specialized heads. Investigated the impact of architectural choices like **Batch Normalization** and **Dropout**, evaluate different **Transfer Learning** strategies, and address class imbalance using **weighted loss functions** to achieve robust "in-the-wild" generalization.

---

##  Experimental Analysis

### 2.1 Internal Dynamics & Regularization
**Batch Normalization** is critical for convergence; it stabilizes internal covariate shift, allowing for a higher maximum stable learning rate ($1e-4$) and significantly faster training. By forcing activations into a stable, zero-centered distribution, prevented gradient explosion in the deep layers of the VGG11 backbone.

### 2.2 The Generalization Gap (Dropout)
Custom Dropout layers act as a key regularizer to bridge the generalization gap. While **No Dropout ($p=0.0$)** leads to rapid training loss descent but poor validation stability, **Dropout ($p=0.5$)** forces the network to learn redundant, robust features. This prevents the 4096-dimensional fully connected layers from memorizing the dataset, ensuring the model generalizes to unseen validation data.

[Image of dropout neural network]

### 2.3 Transfer Learning Showdown
Comparing strategies revealed that **Full Fine-Tuning** outperforms Strict Feature Extraction. While freezing the backbone is efficient, segmentation requires spatial awareness that classification-only weights lack. Unfreezing the final convolutional blocks allows the model to adapt generic edge detectors into dense, semantic spatial maps.

### 2.4 Feature Map Visualization: Edges to Semantics
Analysis of the feature maps shows a clear transition: early layers act as localized edge and texture detectors, while deeper layers (Block 5) collapse spatial resolution into high-level semantic "heatmaps." These deep maps ignore background noise to focus exclusively on class-specific features like ears, snouts, and eyes.

code snippet: https://github.com/Vasumathi9823/DA6401-Assignement-II---EE21D063/blob/main/Code_snippet_2.4_2.6_2.7.py

### 2.5 Object Detection: Confidence & IoU
The model achieves high **Intersection over Union (IoU)** across the test set, though it struggles with low-contrast edge cases (e.g., dark pets against dark backgrounds). These failure cases highlight the model's reliance on clear boundary gradients for accurate coordinate regression.

### 2.6 Segmentation: Dice vs. Pixel Accuracy
Due to heavy class imbalance (~75% background), **Pixel Accuracy** provides a deceptive metric for success. Prioritized the **Dice Coefficient**, which ignores true negatives. The Dice formula is defined as:

$$Dice = \frac{2 \cdot |A \cap B|}{|A| + |B|}$$

By using **Weighted Cross-Entropy**, penalized the background over-fitting and force the U-Net to focus on the minority "Pet" and "Border" classes.

code snippet: https://github.com/Vasumathi9823/DA6401-Assignement-II---EE21D063/blob/main/Code_snippet_2.4_2.6_2.7.py

### 2.7 & 2.8 Meta-Analysis & Reflection
The final pipeline demonstrates strong generalization to "in-the-wild" images. The synergy between the shared encoder and multi-task heads allows the model to balance semantic richness for classification with spatial precision for segmentation. The integration of **Batch Normalization, Strategic Dropout, and Class-Weighted Loss** forms the backbone of this robust vision system.

code snippet: https://github.com/Vasumathi9823/DA6401-Assignement-II---EE21D063/blob/main/Code_snippet_2.4_2.6_2.7.py


---
*All experiments were tracked and logged via Weights & Biases (W&B):*
[**Vasumathi DA6401 Assignment 2 Report**](https://api.wandb.ai/links/ee21d063-iit-madras/3nzqtc5r)
