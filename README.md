# Which-Way-Up

# Face Orientation Classifier

## Project Background


This project addresses the problem of estimating face orientation using machine learning. Four orientation classes were defined: upright, rotated left, rotated right, and upside-down. The work builds on advances in computer vision and deep learning, proposing a practical solution that could be integrated into face detection systems for applications in surveillance and VR. The core objective was to assess the effectiveness of neural networks in this classification task using images of varied resolutions.

## Executive Summary

Developed a PCA + MLP pipeline to classify face orientations (upright, left, right, upside-down). Achieved up to 98% accuracy on high-res images (90×90). Performance dropped at lower resolutions, but hyperparameter tuning improved results across the board.


## Analysis

The dataset was derived from the Labeled Faces in the Wild collection. Images were processed by extracting sub-images of 30×30, 50×50, and 90×90 pixels using a patch extractor. Each sub-image was randomly rotated 0–3 times (in 90° increments) to simulate different face orientations and generate class labels.
Before feeding into the classifier, Principal Component Analysis (PCA) was applied to reduce dimensionality and retain the most important features, improving computational efficiency and model performance.


Five classifiers (KNN, SVM, Random Forest, MLP) were tested. While SVM showed slightly better performance, MLP was chosen due to its faster training and flexibility in tuning.

Key hyperparameters tuned:
 - Hidden layers: 2 hidden layers worked better for low-res images; 1 layer for high-res.
 - Activation functions: ReLU and Tanh outperformed Identity and Logistic.
 - Regularization (alpha): Higher alpha (0.1) improved generalization and reduced overfitting.

Final Model Performance:

Resolution	Accuracy	Precision	Recall
30×30	58.6%	58.5%	58.1%
50×50	80.5%	80.5%	80.5%
90×90	98.1%	98.0%	98.0%

Lower resolutions led to higher misclassification due to lack of features, but overall, the MLP showed strong adaptability. Further tuning and additional hyperparameters could enhance performance even more.


## Technology Used



## Credits

- Dataset by 
- Assignment for COM6018 - Data Science with Python (The University of Sheffield)
- Analysis by Sharath Devanand