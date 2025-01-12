"""
Visualization module for Employee Attrition prediction.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

def plot_feature_importance(model, feature_names, top_n=10, save_path=None):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        top_n (int): Number of top features to display
        save_path (str, optional): Path to save the plot
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    plt.figure(figsize=(10, 6))
    importances.nlargest(top_n).plot(kind='barh')
    plt.title(f'Top {top_n} Most Important Features')
    plt.xlabel('Feature Importance')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path (str, optional): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_curves(models, X_test, y_test, save_path=None):
    """
    Plot ROC curves for multiple models.
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_correlation_matrix(df, save_path=None):
    """
    Plot correlation matrix for numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        save_path (str, optional): Path to save the plot
    """
    corr = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_categorical_distribution(df, column, save_path=None):
    """
    Plot distribution of categorical variables with respect to target.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to plot
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate percentages
    attrition_pct = df.groupby(column)['Attrition'].mean() * 100
    
    # Create bar plot
    ax = attrition_pct.plot(kind='bar')
    plt.title(f'Attrition Rate by {column}')
    plt.xlabel(column)
    plt.ylabel('Attrition Rate (%)')
    
    # Add percentage labels on top of bars
    for i, v in enumerate(attrition_pct):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
