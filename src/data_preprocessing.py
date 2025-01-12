"""
Data preprocessing module for Employee Attrition prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json
import os

def load_data(file_path):
    """
    Load the employee attrition dataset.
    
    Args:
        file_path (str): Path to the dataset
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the data by handling missing values, encoding categorical variables,
    and scaling numerical features.
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        tuple: (X, y, feature_names, categorical_columns, numerical_columns)
    """
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Create target variable based on STATUS and termreason_desc
    data['Attrition'] = (data['STATUS'] != 'ACTIVE').astype(int)
    
    # Select relevant features
    feature_cols = [
        'age', 'length_of_service', 'gender_short', 
        'department_name', 'job_title', 'store_name',
        'city_name', 'BUSINESS_UNIT'
    ]
    X = data[feature_cols].copy()  # Use .copy() to avoid SettingWithCopyWarning
    y = data['Attrition']
    
    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Handle missing values
    for col in numerical_columns:
        X.loc[:, col] = X[col].fillna(X[col].median())
    for col in categorical_columns:
        X.loc[:, col] = X[col].fillna(X[col].mode()[0])
    
    # Handle categorical variables
    le = LabelEncoder()
    for column in categorical_columns:
        X.loc[:, column] = le.fit_transform(X[column])
    
    # Scale numerical features
    scaler = StandardScaler()
    X.loc[:, numerical_columns] = scaler.fit_transform(X[numerical_columns])
    
    return X, y, X.columns, categorical_columns, numerical_columns

def save_processed_data(X, y, feature_names, categorical_columns, numerical_columns, output_dir):
    """
    Save processed data and metadata to files.
    
    Args:
        X: Features DataFrame
        y: Target Series
        feature_names: List of feature names
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
        output_dir: Directory to save the files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features and target
    processed_df = pd.DataFrame(X, columns=feature_names)
    processed_df['Attrition'] = y
    processed_df.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
    
    # Save metadata
    metadata = {
        'feature_names': feature_names.tolist(),
        'categorical_columns': categorical_columns.tolist(),
        'numerical_columns': numerical_columns.tolist()
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of dataset to include in the test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

if __name__ == "__main__":
    # Example usage
    file_path = "../data/raw/employee_attrition.csv"
    
    # Load and preprocess data
    df = load_data(file_path)
    X, y, feature_names, cat_cols, num_cols = preprocess_data(df)
    
    # Save processed data
    output_dir = "../data/processed"
    save_processed_data(X, y, feature_names, cat_cols, num_cols, output_dir)
    
    # Split data
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    
    print("Data preprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
