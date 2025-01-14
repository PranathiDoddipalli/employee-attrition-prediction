# Employee Attrition Analysis Report

## Executive Summary

This report presents the findings from our analysis of employee attrition using machine learning techniques. We developed predictive models to identify factors contributing to employee turnover and provide actionable insights for reducing attrition.

## Data Overview

The dataset contains employee information including:
- Demographics (age, gender)
- Job-related factors (department, job title, length of service)
- Location information (city, store)
- Business unit information

## Key Findings

### 1. Model Performance

We compared three different machine learning models:

1. XGBoost (Best Performing):
   - Accuracy: 98.81%
   - Precision: 96.86%
   - Recall: 62.29%
   - F1 Score: 75.82%
   - ROC AUC: 93.14%

2. Random Forest:
   - Accuracy: 98.63%
   - Precision: 90.45%
   - Recall: 60.61%
   - F1 Score: 72.58%
   - ROC AUC: 90.40%

3. Logistic Regression:
   - Accuracy: 97.01%
   - ROC AUC: 69.71%
   - Note: The logistic regression model had convergence issues and requires further optimization

### 2. Key Insights

1. Model Performance:
   - XGBoost outperformed other models across all metrics
   - High precision (96.86%) indicates very few false positives
   - Lower recall (62.29%) suggests some attrition cases are missed

2. Visualization Insights:
   - Feature importance plots show the most significant factors in predicting attrition
   - ROC curves demonstrate strong discriminative ability of our models
   - Categorical distribution plots reveal patterns in attrition across different groups

## Technical Details

The analysis pipeline includes:
1. Data preprocessing:
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling
2. Model training and evaluation
3. Visualization generation

All code and visualizations are available in the project repository:
- Source code: `/src/`
- Visualizations: `/reports/figures/`
- Processed data: `/data/processed/`


