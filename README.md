# DSCI 552 – Homework 6: Tree-Based Methods and Class Imbalance

Predicting air pressure system (APS) failures in Scania trucks with the UCI [APS Failure](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks) data set: 60,000 training and 16,000 test trucks, 170 sensor features, and only 1.7% positive cases.

## Contents

- Median imputation (99% of rows have missing values), coefficient of variation, correlation analysis, and visualization
- Random forests with and without class weighting, with out-of-bag error estimates
- L1-penalized XGBoost with α chosen by stratified 5-fold cross-validation
- SMOTE oversampling inside the cross-validation folds (the right way to cross-validate with resampling)
- ISLR Exercises 6.6.3, 6.6.5, 8.4.5, and 9.7.3, plus extra practice 9.7.2 and 8.4.4

## Results summary (test set, 375 failures among 16,000 trucks)

| Model | Missed failures | False alarms | Misclassification | AUC |
|---|---|---|---|---|
| Random forest | 102 | 16 | 0.0074 | 0.994 |
| XGBoost L1 | 89 | 20 | 0.0068 | 0.996 |
| XGBoost L1 + SMOTE | **46** | 165 | 0.0132 | 0.994 |

SMOTE halves the number of missed failures at the cost of more false alarms. Using the data set's cost metric (500 per missed failure, 10 per false alarm), it lowers the total cost from 44,700 to 24,650.

## How to run

1. Download and unzip the APS data inside the repo folder (the data is not included).
2. Install the dependencies: `pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn`
3. Open `HW6.ipynb` and run all cells. The SMOTE cross-validation takes several minutes.
