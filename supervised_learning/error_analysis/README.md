# Error Analysis

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project focuses on error analysis in machine learning by calculating various metrics from confusion matrices, including sensitivity, precision, specificity, and F1 score. It also addresses common scenarios in model performance and provides strategies for managing bias and variance. Finally, it includes a comparative analysis of confusion matrices to identify significant performance issues.

## Prerequisites

- Python 3.x
- NumPy
- Pandas (optional, for data handling)
- Scikit-learn (optional, for additional metrics)

## Task Summaries

0. **Create Confusion**:  
   Generates a confusion matrix from one-hot encoded true labels and predicted logits using NumPy.

1. **Sensitivity**:  
   Calculates the sensitivity for each class in a confusion matrix, returning an array of sensitivity values.

2. **Precision**:  
   Computes the precision for each class in a confusion matrix, providing an array of precision values.

3. **Specificity**:  
   Determines the specificity for each class in a confusion matrix and returns an array of specificity values.

4. **F1 Score**:  
   Calculates the F1 score for each class in a confusion matrix, utilizing previously computed sensitivity and precision values.

5. **Dealing with Error**:  
   Outlines strategies for addressing different bias and variance scenarios, formatted as a list for multiple approaches.

6. **Compare and Contrast**:  
   Analyzes provided training and validation confusion matrices to identify the most pressing issue, categorizing it as high bias, high variance, or nothing.
