# Augmented_Dataset_ML_Training_CS203_LAB05
# Image Classification with Data Augmentation using Augly and ResNet-50

This project demonstrates image classification of cats and dogs using the ResNet-50 architecture and data augmentation techniques with the Augly library. The goal is to improve model performance by generating variations of training images, making the model more robust and less prone to overfitting.

## Introduction

Image classification is a fundamental task in computer vision, aiming to categorize images into predefined classes.  This project tackles the classic cat vs. dog classification problem.  We leverage the power of transfer learning by using a pre-trained ResNet-50 model. ResNet-50 is a deep convolutional neural network known for its strong performance in image recognition tasks.  Transfer learning allows us to fine-tune the model on our specific dataset, leveraging knowledge gained from a larger, more general dataset (like ImageNet).  A key aspect of this project is the use of data augmentation. Augly, a powerful Python library, is used to generate variations of the training images, such as rotations, flips, and slight color adjustments. This artificially increases the size and diversity of the training data, helping the model generalize better to unseen images and prevent overfitting, which occurs when a model performs well on the training data but poorly on new, unseen data.

## Dataset

The dataset used in this project consists of images of cats and dogs. The data used in our demo is [Cats and Dogs image classification] by [Samuel Cortinhas], from kaggle. The data is organized into directories, with separate folders for training, augmented training (containing the augmented images), and testing sets. Each of these sets contains subfolders for "cats" and "dogs." The project also includes commented-out code showing how to download the dataset from Kaggle Hub, if needed, otherwise the file is pre-downloaded in the main directory as well.


## Model Architecture

This project utilizes the ResNet-50 architecture, a variant of the Deep Residual Network (ResNet) family. ResNet models were introduced to address the challenge of training increasingly deep neural networks.  As networks become deeper, they can suffer from the "degradation" problem, where accuracy saturates and even declines. This isn't due to overfitting, but rather difficulties in optimization, potentially related to vanishing/exploding gradients.

ResNet-50, specifically, is a 50-layer deep convolutional neural network. It leverages the core concept of "residual learning" through the use of "skip connections" or "shortcut connections." These connections allow the network to learn "identity mappings," meaning they can effectively pass the input of a layer directly to its output if that proves to be the optimal operation.  This helps to alleviate the degradation problem and makes it easier to train very deep networks.

![Model Architecture] (Architecture.jpg)
![How it makes the process fast] (Skip_layer.jpg)

Crucially, the code provided initializes the ResNet-50 model with no pre-trained weights (weights=None). This means the model starts with randomly initialized weights, not weights pre-trained on a large dataset like ImageNet.  This is a significant difference.  While transfer learning (using pre-trained weights) is common with ResNet-50, this project trains the model from scratch on the cats vs. dogs dataset.

The code then modifies the final fully connected layer (model.fc = nn.Linear(model.fc.in_features, 2)) to have two output units, corresponding to the two classes (cat and dog).  This adjustment is necessary because the original output layer of ResNet-50 is designed for a much larger number of classes (e.g., 1000 for ImageNet).

Therefore, the ResNet-50 model in this project is not being used in a transfer learning setup. Instead, it is being trained from scratch on the cats vs. dogs dataset.  This is important to note, as training from scratch typically requires more data and computational resources compared to fine-tuning a pre-trained model.  It also means the model's performance will depend heavily on the size and quality of the training data, making data augmentation even more critical in this scenario.


## Training Process

The model is trained using the Adam optimizer and the Cross-Entropy Loss function.  The Adam optimizer is a popular choice for training deep learning models due to its efficiency and adaptability. The Cross-Entropy Loss is a standard loss function for multi-class classification problems.  The training process involves iterating over the training data in batches for a certain number of epochs.  During each epoch, the model makes predictions on the training images, calculates the loss, and updates its weights to minimize the loss.  The training is performed twice: once on the original training dataset and again on the augmented training dataset.  This allows us to compare the performance of the model with and without data augmentation.

## Evaluation Metrics

The performance of the trained models is evaluated on a held-out test set using the following metrics:

*   **Precision:**  Out of all the images predicted as a certain class (e.g., 'cat'), how many were actually correct?
*   **Recall:** Out of all the images that belong to a certain class (e.g., 'cat'), how many did the model correctly identify?
*   **F1 Score:** The harmonic mean of precision and recall, providing a balanced measure of accuracy.
*   **Accuracy:** The overall percentage of correctly classified images.

## Results and Analysis

The results demonstrate the positive impact of data augmentation. The model trained on the augmented dataset shows a significant improvement in recall, indicating that it is much better at correctly identifying actual cats and dogs.  While precision may slightly decrease, the overall F1 score and accuracy improve, showing that the model trained with data augmentation generalizes better to new, unseen images.  This confirms the effectiveness of using Augly to create more diverse training data.

## Dataset Visualization

A bar graph is generated to visualize the number of cat and dog images in each of the training, augmented training, and test sets. This visualization provides insight into the class distribution within the datasets, which is important for understanding potential biases or imbalances.  This graph helps to ensure there is a reasonable distribution of images in each class.

## Conclusion

This project successfully demonstrates the use of data augmentation with the Augly library to improve the performance of a ResNet-50 model for cat vs. dog image classification.  The results highlight the importance of data augmentation in enhancing model robustness and generalization.  This approach can be applied to other image classification tasks to achieve similar improvements.
