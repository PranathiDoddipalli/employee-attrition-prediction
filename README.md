# Employee Attrition Prediction

A comprehensive machine learning project to predict employee attrition using the Kaggle Employee Attrition Dataset. This project demonstrates skills in data analytics, predictive modeling, and business insights generation.

## Project Overview

This project aims to create an end-to-end solution for predicting employee attrition, helping organizations identify potential turnover risks and implement targeted retention strategies.

## Project Structure

```
employee_attrition/
├── data/               # Data files
│   ├── raw/           # Original dataset
│   └── processed/     # Cleaned and processed data
├── notebooks/         # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_development.ipynb
├── src/              # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── visualization.py
├── models/           # Saved model files
├── reports/          # Generated analysis reports
└── requirements.txt  # Project dependencies
```

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place it in the `data/raw/` directory

## Project Components

1. **Data Exploration and Cleaning**
   - Exploratory Data Analysis (EDA)
   - Missing value handling
   - Data visualization

2. **Feature Engineering**
   - Categorical variable encoding
   - Feature scaling
   - Feature selection

3. **Model Development**
   - Model comparison (Logistic Regression, Random Forest, XGBoost)
   - Model evaluation metrics
   - Hyperparameter optimization

4. **Visualization and Interpretation**
   - Feature importance plots
   - Confusion matrices
   - ROC curves

5. **Business Insights**
   - Key findings
   - Actionable recommendations

## License

This project is licensed under the MIT License.
