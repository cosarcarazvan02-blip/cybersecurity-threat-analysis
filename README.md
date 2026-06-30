# Cybersecurity Threat Detection

A machine learning project focused on analyzing and classifying cybersecurity threat alerts. The code in main.py includes exploratory data analysis, full preprocessing pipelines, baseline model benchmarking, and final model optimization.

## Core Features

* Automated preprocessing via ColumnTransformer: mean imputation and scaling for numeric features, and most-frequent imputation with One-Hot Encoding for categorical features.
* Real-world data simulation: artificial injection of a 5% missingness rate (NaN) on numeric variables to test the robustness of the imputation step.
* Class imbalance handling: integration of SMOTE (imbalanced-learn) directly into the training pipeline.
* Hyperparameter tuning: 5-fold cross-validation Grid Search to find the optimal parameters for the Random Forest classifier.
* Advanced evaluation: learning curve generation to check for overfitting, and confusion matrices normalized in three different ways.

## Requirements and Installation

To run this script, you need the following libraries:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn imbalanced-learn
