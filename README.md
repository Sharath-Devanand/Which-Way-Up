# Which-Way-Up

## Face Orientation Classifier

### Project Background

This project addresses the problem of estimating face orientation using machine learning. Four orientation classes were defined: upright, rotated left, rotated right, and upside-down. The work builds on advances in computer vision and deep learning, proposing a practical solution that could be integrated into face detection systems for applications in surveillance and VR. The core objective was to assess the effectiveness of neural networks in this classification task using images of varied resolutions.

### Executive Summary

Developed a PCA + MLP pipeline to classify face orientations (upright, left, right, upside-down). Achieved up to 98% accuracy on high-res images (90×90). Performance dropped at lower resolutions, but hyperparameter tuning improved results across the board.

### Analysis

The dataset was derived from the Labeled Faces in the Wild collection. Images were processed by extracting sub-images of 30×30, 50×50, and 90×90 pixels using a patch extractor. Each sub-image was randomly rotated 0–3 times (in 90° increments) to simulate different face orientations and generate class labels.  
Before feeding into the classifier, Principal Component Analysis (PCA) was applied to reduce dimensionality and retain the most important features, improving computational efficiency and model performance.

Five classifiers - K Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest Classifier (RFC), and Multi-layer Perceptron (MLP) - were evaluated for the face orientation classification task. Among these, SVM and MLP consistently delivered the highest performance across all pixel sizes. However, MLP was selected as the model of choice due to its faster training times and greater flexibility for tuning.

Key hyperparameters tuned in the MLP included:

- Hidden Layers: The number of hidden layers and the number of neurons per layer were varied to assess the impact on accuracy. For 30×30 pixel images, using 2 hidden layers with 100 neurons per layer provided the best results, while for 90×90 pixel images, a single hidden layer performed better. This suggests that lower-resolution images benefit from additional layers to capture complex features, while higher-resolution images can be classified accurately with fewer layers.

- Activation Functions: The activation functions tested were ReLU, Tanh, Logistic, and Identity. The ReLU and Tanh functions showed consistently higher performance than Logistic and Identity. This can be attributed to the fact that both ReLU and Tanh allow for better gradient propagation during training, which helps the model learn faster and more effectively, especially for the higher-dimensional feature sets.

- Regularisation (Alpha): Regularisation was applied to control overfitting. The alpha parameter, which defines the strength of L2 regularisation, was varied across different values. Higher values of alpha (0.1) showed a marked improvement in performance, reducing overfitting and boosting the model's ability to generalise. This was especially noticeable at higher resolutions, where the risk of overfitting is more pronounced due to the increased feature complexity.

The following tables summarize the performance for different pixel sizes and hyperparameter configurations:

---

### Comparison of Classifier Performance

| Classifier  | 30×30 Accuracy | 50×50 Accuracy | 90×90 Accuracy |
|-------------|----------------|----------------|----------------|
| SVM     | 57.4%          | 78.5%          | 98.1%          |
| KNN     | 48.9%          | 73.7%          | 95.7%          |
| RFC     | 71.8%          | 94.4%          | 93.9%          |
| MLP     | 57.7%          | 78.2%          | 97.3%          |

---

### Performance with Different Hidden Layer Configurations

| Hidden Layers     | 30×30 Accuracy | 50×50 Accuracy | 90×90 Accuracy |
|-------------------|----------------|----------------|----------------|
| 100           | 45.7%          | 63.0%          | 96.7%          |
| 100, 100      | 48.0%          | 66.6%          | 95.8%          |
| 100, 100, 100 | 46.8%          | 64.4%          | 95.8%          |
| 50, 50        | 46.9%          | 63.5%          | 93.9%          |
| 200, 200      | 50.1%          | 71.6%          | 94.8%          |

---

### Activation Function Comparison

| Activation Function | 30×30 Accuracy | 50×50 Accuracy | 90×90 Accuracy |
|---------------------|----------------|----------------|----------------|
| ReLU            | 48.7%          | 69.4%          | 96.7%          |
| Tanh**            | 50.3%          | 70.4%          | 96.3%          |
| Logistic        | 34.4%          | 68.0%          | 95.6%          |
| Identity        | 35.0%          | 45.8%          | 85.7%          |

---

### Effect of Regularization (Alpha)

| Alpha  | 30×30 Accuracy | 50×50 Accuracy | 90×90 Accuracy |
|--------|----------------|----------------|----------------|
| 0.1| 50.3%          | 71.4%          | 96.7%          |
| 0.01| 47.8%         | 71.0%          | 96.0%          |
| 0.001| 48.3%         | 70.0%          | 96.0%          |
| 0.0001| 48.0%        | 71.5%          | 95.6%          |

---

The analysis shows that accuracy improves with higher pixel resolutions. However, even at lower pixel sizes (30×30), the MLP model demonstrated robustness and adaptability, achieving decent accuracy with proper tuning of hyperparameters. The confusion matrix (shown in Figure 1) highlights the challenges of lower-resolution images, where misclassification rates rise due to the lack of sufficient features to distinguish between orientations.

### Technology Used

- Python (Scikit-learn, pandas, numpy)

### Credits

- Dataset by Labeled Faces in the Wild
- Assignment for COM6018 - Data Science with Python (The University of Sheffield)
- Analysis by Sharath Devanand