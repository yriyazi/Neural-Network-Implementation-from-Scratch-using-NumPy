# Utils

The "utils" folder contains utility functions that support various aspects of the neural network implementation and its usage.

## Table of Contents

- [Introduction](#introduction)
- [Functions](#functions)
  - [`compute_confusion_matrix(true, pred)`](#compute_confusion_matrix)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The "utils" folder within this repository contains utility functions that are helpful for different tasks related to the neural network implementation. These functions provide support for common operations, such as evaluating model performance, data preprocessing, and more.

## Functions

  ### `compute_confusion_matrix(true, pred)`
    
    Computes a confusion matrix using NumPy for two numpy arrays: `true` and `pred`.
    
    Args:
    - `true` (numpy.ndarray): Array of true labels.
    - `pred` (numpy.ndarray): Array of predicted labels.
    
    Returns:
    - numpy.ndarray: The confusion matrix where rows represent true classes and columns represent predicted classes.
    
    This function calculates a confusion matrix to evaluate the performance of a classification model. It counts the occurrences of true positive, true negative, false positive, and false negative predictions for each class. The result is a matrix that provides insights into the model's classification accuracy and misclassifications across different classes. This implementation is independent of the scikit-learn library, providing an alternative way to compute confusion matrices.


  ### `convertToOneHot(vector,num_classes=None)`
    returns a one hotted vector accoding to num_classes

## Contributing

Contributions to the utility functions in the "utils" folder are welcome! If you have additional utility functions to add or improvements to suggest, please feel free to open a pull request.

## License

The utility functions in the "utils" folder are part of the overall project and are licensed under the same [MIT License](../LICENSE) as the main repository.

---

*Disclaimer: The utility functions provided here are for educational and illustrative purposes, and may not cover all scenarios or optimizations for production-level use.*
