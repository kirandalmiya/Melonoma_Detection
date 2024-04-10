# Melanoma Detection using Convolutional Neural Network (CNN)

## Table of Contents

* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Project Pipeline](#project-pipeline)
* [Conclusion](#conclusion)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information
In this project, we aim to build a multiclass classification model using a custom convolutional neural network (CNN) in TensorFlow to accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early, accounting for 75% of skin cancer deaths. By developing a solution that can evaluate images and alert dermatologists about the presence of melanoma, we aim to reduce manual effort needed in diagnosis.

The dataset consists of 2357 images of malignant and benign oncological diseases, sourced from the International Skin Imaging Collaboration (ISIC). The dataset contains images of various skin diseases, including
1. Actinic keratosis
2. Basal cell carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented benign keratosis
7. Seborrheic keratosis
8. Squamous cell carcinoma
9. Vascular lesion


## Technologies Used

- TensorFlow - version 2.16.1

- Python version: 3.11.5

- Matplotlib - version 3.7.2

- NumPy - version1.24.3

- Augmentor - version 0.2.12

## Project Pipeline

1. **Data Reading/Data Understanding**: Define the path for train and test images and explore the dataset.
2. **Dataset Creation**: Create train and validation datasets from the train directory with a batch size of 32. Resize images to 180x180.
3. **Dataset Visualization**: Visualize one instance of all nine classes present in the dataset.
4. **Model Building & Training**: 
   - Create a CNN model to accurately detect the nine classes in the dataset.
   - Rescale images to normalize pixel values between 0 and 1.
   - Choose an appropriate optimizer and loss function for model training.
   - Train the model for approximately 20 epochs and analyze findings for evidence of overfitting or underfitting.
5. **Data Augmentation & Model Building**:
   - Implement a data augmentation strategy to resolve underfitting/overfitting.
   - Build and train the CNN model on augmented data.
   - Evaluate if the earlier issue is resolved.
6. **Class Distribution Analysis**:
   - Examine the current class distribution in the training dataset.
   - Identify classes with the least number of samples and those dominating the data.
7. **Handling Class Imbalances**:
   - Rectify class imbalances in the training dataset using the Augmentor library.
8. **Model Building & Training on Rectified Data**:
   - Train the CNN model on the rectified class imbalance data.
   - Analyze findings to see if the issues are resolved.

## Conclusion
Model 1: Around the 19th and 20th epochs, we see a difference in performance between training and validation data. This indicates overfitting, where the model is too focused on the training data and struggles with new data.

Training Accuracy: 85.04 %
Validation Accuracy: 49.44%

Model 2 with augmented data: 
Training accuracy has not increased when compared to the basic model, 
Although there hasn't been improvement in accuracy(Its decresed actually), the implementation of data augmentation has effectively addressed the overfitting issue.

Considering that the model hasn't been trained for a sufficient number of epochs, it's premature to draw any conclusions. Increasing the number of epochs might lead to improved accuracy.

In summary, while data augmentation has mitigated overfitting, further training could potentially enhance the model's accuracy.

Training Accuracy: 48.41
Validation Accuracy: 52.35

Model 3:
-Accuracy on training data has increased by using Augmentor library

-Model is still overfitting

-The problem of overfitting can be solved by add more layer,neurons or adding dropout layers.

-The Model can be further improved by tuning the hyperparameter

Training Accuracy: 91.62
Validation Accuracy: 79.58

Final Model after rectifing class imbalance: Because to augmentation and class imbalance management, training and validation accuracy has enhanced dramatically. The model does not overfit and is performing resonable well. This model can serve as the final model.

Training Accuracy: 89.68%
Validation Accuracy: 84.46%

## Acknowledgements

This project was driven by the imperative to create an automated solution for melanoma detection, aiming to enhance early diagnosis and treatment. We express our gratitude to the International Skin Imaging Collaboration (ISIC) for generously providing access to their dataset, which was pivotal in our research efforts.

Furthermore, we acknowledge the invaluable insights gained from UpGrad live sessions and UpGrad tutorials focused on Convolutional Neural Networks (CNNs). These educational resources served as guiding lights, imparting essential knowledge and expertise that greatly facilitated the successful execution of this assignment.

## Contact
Created by [@kirandalmiya]  (https://github.com/kirandalmiya)
