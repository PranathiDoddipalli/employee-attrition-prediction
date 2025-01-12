import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Load data and models
data = pd.read_csv('data/processed/processed_data.csv')
xgb_model = joblib.load('models/xgboost.joblib')

# Create directory for additional visualizations
import os
os.makedirs('reports/figures/detailed', exist_ok=True)

# 1. Attrition Rate by Department Size
plt.figure(figsize=(15, 8))
dept_stats = data.groupby('department_name').agg({
    'Attrition': ['count', 'mean']
}).reset_index()
dept_stats.columns = ['Department', 'Size', 'Attrition_Rate']
dept_stats['Attrition_Rate'] *= 100

plt.scatter(dept_stats['Size'], dept_stats['Attrition_Rate'], 
           alpha=0.6, s=100)
for i, row in dept_stats.iterrows():
    plt.annotate(f"Dept {i}", 
                (row['Size'], row['Attrition_Rate']),
                xytext=(5, 5), textcoords='offset points')

plt.xlabel('Department Size (Number of Employees)')
plt.ylabel('Attrition Rate (%)')
plt.title('Attrition Rate vs Department Size')
plt.savefig('reports/figures/detailed/dept_size_attrition.png', bbox_inches='tight', dpi=300)
plt.close()

# 2. Age Distribution by Attrition
plt.figure(figsize=(12, 6))
sns.kdeplot(data=data[data['Attrition']==0], x='age', label='Retained', fill=True)
sns.kdeplot(data=data[data['Attrition']==1], x='age', label='Left', fill=True)
plt.title('Age Distribution: Retained vs Left Employees')
plt.xlabel('Age (Standardized)')
plt.ylabel('Density')
plt.savefig('reports/figures/detailed/age_distribution.png', bbox_inches='tight', dpi=300)
plt.close()

# 3. Service Length vs Attrition
plt.figure(figsize=(12, 6))
sns.boxplot(x='Attrition', y='length_of_service', data=data)
plt.title('Length of Service Distribution by Attrition Status')
plt.xlabel('Attrition Status (0=Retained, 1=Left)')
plt.ylabel('Length of Service (Standardized)')
plt.savefig('reports/figures/detailed/service_length_box.png', bbox_inches='tight', dpi=300)
plt.close()

# 4. Business Unit Analysis
plt.figure(figsize=(15, 6))
bu_attrition = data.groupby('BUSINESS_UNIT')['Attrition'].agg(['count', 'mean']).reset_index()
bu_attrition['mean'] *= 100
bu_attrition = bu_attrition.sort_values('mean', ascending=True)

plt.barh(bu_attrition['BUSINESS_UNIT'], bu_attrition['mean'])
plt.xlabel('Attrition Rate (%)')
plt.ylabel('Business Unit')
plt.title('Attrition Rate by Business Unit')
for i, v in enumerate(bu_attrition['mean']):
    plt.text(v + 0.1, i, f'{v:.1f}%')
plt.savefig('reports/figures/detailed/business_unit_attrition.png', bbox_inches='tight', dpi=300)
plt.close()

# 5. Location Analysis
plt.figure(figsize=(15, 8))
city_attrition = data.groupby('city_name')['Attrition'].agg(['count', 'mean']).reset_index()
city_attrition['mean'] *= 100
city_attrition = city_attrition[city_attrition['count'] > 100]  # Filter for cities with >100 employees
city_attrition = city_attrition.sort_values('mean', ascending=True)

plt.barh(city_attrition['city_name'], city_attrition['mean'])
plt.xlabel('Attrition Rate (%)')
plt.ylabel('City')
plt.title('Attrition Rate by City (Cities with >100 Employees)')
for i, v in enumerate(city_attrition['mean']):
    plt.text(v + 0.1, i, f'{v:.1f}%')
plt.savefig('reports/figures/detailed/city_attrition.png', bbox_inches='tight', dpi=300)
plt.close()

# Print summary statistics
print("Detailed Analysis Results")
print("-" * 50)

print("\n1. Department Size vs Attrition Correlation:")
correlation = np.corrcoef(dept_stats['Size'], dept_stats['Attrition_Rate'])[0,1]
print(f"Correlation coefficient: {correlation:.3f}")

print("\n2. Age Statistics by Attrition:")
age_stats = data.groupby('Attrition')['age'].describe()
print(age_stats)

print("\n3. Length of Service Analysis:")
service_stats = data.groupby('Attrition')['length_of_service'].describe()
print(service_stats)

print("\n4. Business Unit Summary:")
print(bu_attrition.to_string(index=False))

print("\n5. Top 5 Cities by Attrition Rate (min 100 employees):")
print(city_attrition.head().to_string(index=False))

# Save summary to file
with open('reports/detailed_analysis.txt', 'w') as f:
    f.write("Detailed Analysis Results\n")
    f.write("-" * 50 + "\n")
    
    f.write("\n1. Department Size vs Attrition Correlation:\n")
    f.write(f"Correlation coefficient: {correlation:.3f}\n")
    
    f.write("\n2. Age Statistics by Attrition:\n")
    f.write(age_stats.to_string())
    
    f.write("\n\n3. Length of Service Analysis:\n")
    f.write(service_stats.to_string())
    
    f.write("\n\n4. Business Unit Summary:\n")
    f.write(bu_attrition.to_string())
    
    f.write("\n\n5. Top 5 Cities by Attrition Rate (min 100 employees):\n")
    f.write(city_attrition.head().to_string())
