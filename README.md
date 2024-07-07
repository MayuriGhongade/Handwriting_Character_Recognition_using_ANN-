# Handwritten Character Recognition using EMNIST Dataset

## Overview

This project implements a deep learning model using Python, TensorFlow, and Keras for recognizing handwritten characters from the EMNIST dataset. 

## Dataset

The EMNIST dataset consists of handwritten character digits, converted to a 28x28 pixel image format.

- **Training dataset:** `emnist-balanced-train.csv`
- **Test dataset:** `emnist-balanced-test.csv`
- **Dataset location:** `/kaggle/input/emnist`

## Project Structure

- `model_training_evaluation.ipynb`: Jupyter notebook containing both model training and evaluation.

## Model Architecture

- Convolutional layers with ReLU activation
- Max pooling layers for down-sampling
- Dense layers for classification
- Output layer with softmax activation for multi-class classification

## Results

- **Validation Accuracy:** 90.13%
- **Test Accuracy:** 89.94%

## Visualization

- Sample images with predicted labels
- Training and validation accuracy and loss curves

## Testing on New Data

- Example of predicting a handwritten character from a new image (`B.jpg`)
- Prediction accuracy and visualization of the prediction

## Conclusion

This project successfully implements a deep learning model for handwritten character recognition using the EMNIST dataset. Future improvements could involve exploring more advanced architectures or fine-tuning hyperparameters.

## References

- EMNIST dataset source: [link](https://www.nist.gov/itl/iad/image-group/emnist-dataset)
- TensorFlow documentation: [link](https://www.tensorflow.org/)
