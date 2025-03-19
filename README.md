# Which-Way-Up

# **Face Orientation Classifier**

## **Overview**  
The objective of this project is to build a classifier that can identify the orientation of a square region from a face image. The model must determine whether the image is:
- Right-side up  
- Rotated 90 degrees to the right  
- Rotated 90 degrees to the left  
- Upside down  

The classifier is trained on sub-images of varying sizes (30px, 50px, and 90px) taken from a larger face image (94x125 pixels).

## **Task Description**  
The task is broken down into building **three separate scikit-learn models**:
- **model.30.joblib** for 30px images
- **model.50.joblib** for 50px images
- **model.90.joblib** for 90px images  

### **Workflow**
1. **Train a Model**: Use the provided training data to build a classification model.
2. **Evaluate the Model**: Test the model using the `evaluate.py` script to check its performance on various image sizes.