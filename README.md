# Academic Project, Training CNNs from Scratch with Augmentation and Interpretability 		
Authors: Gaurav Srivastava, Divyansh Saini

# Image Classification with Data Augmentation using Augly and ResNet-50

This project demonstrates image classification of cats and dogs using the ResNet-50 architecture and data augmentation techniques with the Augly library. The goal is to improve model performance by generating variations of training images, making the model more robust and less prone to overfitting.

## Introduction

Image classification is a fundamental task in computer vision, aiming to categorize images into predefined classes.  This project tackles the classic cat vs. dog classification problem.  We leverage the ResNet-50 architecture. ResNet-50 is a deep convolutional neural network known for its strong performance in image recognition tasks.  A key aspect of this project is the use of data augmentation. Augly, a powerful Python library, is used to generate variations of the training images, such as rotations, flips, and slight color adjustments. This artificially increases the size and diversity of the training data, helping the model generalize better to unseen images and prevent overfitting, which occurs when a model performs well on the training data but poorly on new, unseen data.  **Importantly, this project trains the ResNet-50 model *from scratch* (with randomly initialized weights), rather than using pre-trained weights (transfer learning).** This makes data augmentation even more crucial.

## Dataset

The dataset used in this project consists of images of cats and dogs. The data used in our demo is [Cats and Dogs image classification] by [Samuel Cortinhas], from kaggle. The data is organized into directories, with separate folders for training, augmented training (containing the augmented images), and testing sets. Each of these sets contains subfolders for "cats" and "dogs." The project also includes commented-out code showing how to download the dataset from Kaggle Hub, if needed, otherwise the file is pre-downloaded in the main directory as well.

The directory structure is as follows (after unzipping and splitting):
(forward to end of the repo to see it)

## Model Architecture

This project utilizes the ResNet-50 architecture, a variant of the Deep Residual Network (ResNet) family. ResNet models were introduced to address the challenge of training increasingly deep neural networks.  As networks become deeper, they can suffer from the "degradation" problem, where accuracy saturates and even declines. This isn't due to overfitting, but rather difficulties in optimization, potentially related to vanishing/exploding gradients.

ResNet-50, specifically, is a 50-layer deep convolutional neural network. It leverages the core concept of "residual learning" through the use of "skip connections" or "shortcut connections." These connections allow the network to learn "identity mappings," meaning they can effectively pass the input of a layer directly to its output if that proves to be the optimal operation.  This helps to alleviate the degradation problem and makes it easier to train very deep networks.

![Model Architecture](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*VM94wVftxP7wkiKo4BjfLA.png)

*Model Architecture*

![How it makes the process fast](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*WXiXzRpbUtdTpVOHMtQA8w.png)

*How it makes the process faster*

Crucially, the code provided initializes the ResNet-50 model with *no* pre-trained weights (`weights=None`). This means the model starts with randomly initialized weights, not weights pre-trained on a large dataset like ImageNet.  This is a significant difference.  While transfer learning (using pre-trained weights) is common with ResNet-50, this project trains the model *from scratch* on the cats vs. dogs dataset.

The code then modifies the final fully connected layer (`model.fc = nn.Linear(model.fc.in_features, 2)`) to have two output units, corresponding to the two classes (cat and dog).  This adjustment is necessary because the original output layer of ResNet-50 is designed for a much larger number of classes (e.g., 1000 for ImageNet).

Therefore, the ResNet-50 model in this project is *not* being used in a transfer learning setup. Instead, it is being trained from scratch on the cats vs. dogs dataset.  This is important to note, as training from scratch typically requires more data and computational resources compared to fine-tuning a pre-trained model.  It also means the model's performance will depend heavily on the size and quality of the training data, making data augmentation even more critical in this scenario.

## Training Process

The model is trained using the Adam optimizer and the Cross-Entropy Loss function.  The Adam optimizer is a popular choice for training deep learning models due to its efficiency and adaptability. The Cross-Entropy Loss is a standard loss function for multi-class classification problems.  The training process involves iterating over the training data in batches for a certain number of epochs.  During each epoch, the model makes predictions on the training images, calculates the loss, and updates its weights to minimize the loss.  The training is performed twice: once on the original training dataset and again on the augmented training dataset.  This allows us to compare the performance of the model with and without data augmentation.

## Model Parameters

This section outlines the key parameters used for training the image classification model:

| Parameter | Value | Description |
|---|---|---|
| **Model** | ResNet-50 | A deep convolutional neural network architecture. Initialized with *random* weights (not pre-trained). |
| **Batch Size** | 32 | The number of images processed in each training iteration. |
| **Number of Epochs** | 10 | The number of times the model iterates over the entire training dataset. |
| **Learning Rate** | 0.001 |  Controls the step size the model takes during optimization. |
| **Optimizer** | Adam | An optimization algorithm used to update the model's weights during training. |
| **Loss Function** | Cross-Entropy Loss | Measures the difference between the model's predictions and the actual labels. |
| **Data Augmentation** | Augly library | Used to generate variations of the training images to improve model robustness.  The specific augmentations are handled by a separate script and result in the `train_aug` directory. |

These parameters were chosen based on empirical evidence and best practices in image classification.  Training from scratch with a relatively small dataset makes careful parameter selection important.

## Evaluation Metrics

The performance of the trained models is evaluated on a held-out test set using the following metrics:

*   **Precision:**  Out of all the images predicted as a certain class (e.g., 'cat'), how many were actually correct?
*   **Recall:** Out of all the images that belong to a certain class (e.g., 'cat'), how many did the model correctly identify?
*   **F1 Score:** The harmonic mean of precision and recall, providing a balanced measure of accuracy.
*   **Accuracy:** The overall percentage of correctly classified images.

These metrics were calculated on a held-out test set (`/content/split_data/test`) that the model did not see during training. This provides an unbiased estimate of the model's generalization ability.  The `evaluate_model` function in the code calculates and prints these metrics.

## Results and Analysis

The results demonstrate the positive impact of data augmentation. The model trained on the augmented dataset shows a significant improvement in recall, indicating that it is much better at correctly identifying actual cats and dogs.  While precision may slightly decrease, the overall F1 score and accuracy improve, showing that the model trained with data augmentation generalizes better to new, unseen images.  This confirms the effectiveness of using Augly to create more diverse training data.

**Results:**

| Metric | Original Dataset |  Augmented Dataset |
|---|---|---|
|**Precision** |    0.5714 | 0.5789|
|**Recall** |    0.5714     | 0.7857|
|**F1 Score** |    0.5714 |    0.6667|
|**Accuracy** |    0.5714     | 0.6071|

**Analysis:**

The model trained on the original dataset exhibits a baseline performance across all metrics. Precision and recall are identical (0.5714), indicating a balance between identifying true positives and minimizing false positives. However, this balance comes at the cost of a relatively low overall performance, reflected in the F1 score (0.5714) and accuracy (0.5714).

The model trained on the augmented dataset demonstrates a shift in performance characteristics. While precision shows a slight improvement (0.5789), the most notable change is the substantial increase in recall (0.7857). This indicates that the augmented model is significantly better at correctly identifying positive cases (cats or dogs) within the dataset.

The improvement in recall comes with a trade-off, as the augmented model's precision is marginally lower. However, this trade-off is often desirable in scenarios where identifying all positive cases is crucial, even if it means a slight increase in false positives.

The augmented model's F1 score (0.6667) and accuracy (0.6071) demonstrate a clear advantage over the original dataset. The F1 score, being the harmonic mean of precision and recall, highlights the augmented model's better balance in capturing true positives while minimizing false positives and false negatives. The increased accuracy further supports the notion that data augmentation using the Augly library has improved the model's overall ability to correctly classify images of cats and dogs.

## Dataset Visualization

A bar graph is generated to visualize the number of cat and dog images in each of the training, augmented training, and test sets. This visualization provides insight into the class distribution within the datasets, which is important for understanding potential biases or imbalances.  This graph helps to ensure there is a reasonable distribution of images in each class.  The bar graph is generated by the `count_images` and plotting code at the end of the provided Python script.

## Conclusion

This project successfully demonstrates the use of data augmentation with the Augly library to improve the performance of a ResNet-50 model (trained *from scratch*) for cat vs. dog image classification.  The results highlight the importance of data augmentation in enhancing model robustness and generalization, especially when training without pre-trained weights.  This approach can be applied to other image classification tasks to achieve similar improvements.

## Code Overview

The provided Python code performs the following steps:

1.  **Setup:** Installs necessary libraries (Augly), sets the device (CPU or GPU), and defines file paths.
2.  **Hyperparameters:** Defines key training parameters like batch size, epochs, and learning rate.
3.  **Data Transformations:** Defines image transformations using `torchvision.transforms`, including resizing, converting to tensors, and normalization.  These are applied to both the original and augmented datasets.
4.  **Data Loading:**  The `load_dataset` function uses `ImageFolder` to load the images from the specified directories and creates `DataLoader` objects for efficient batch processing.
5.  **Model Initialization:** The `initialize_model` function creates a ResNet-50 model with *random* weights (`weights=None`) and modifies the final fully connected layer for binary classification (2 output classes).
6.  **Training:** The `train_model` function trains the model using the Adam optimizer and Cross-Entropy Loss.  It iterates through epochs and batches, calculating the loss and updating the model's weights.
7.  **Evaluation:** The `evaluate_model` function evaluates the trained model on the test set, calculating and printing precision, recall, F1 score, and accuracy.
8.  **Training and Evaluation Runs:** The code trains and evaluates two models: one on the original dataset and one on the augmented dataset.
9. **Dataset Visualization:** Creates a bar chart to visualize the number of cat and dog images across train, augmented train and test datasets.


```text
/content/
    split_data/
        train/
            cats/
                *.jpg
                ...
            dogs/
                *.jpg
                ...
        train_aug/
            cats/
                *.jpg
                ...
            dogs/
                *.jpg
                ...
        test/
            cats/
                *.jpg
                ...
            dogs/
                *.jpg
                ...
    test.zip
    test/
        cats/
            *.jpg
            ...
        dogs/
            *.jpg
            ...
