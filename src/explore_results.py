import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
processed_data = pd.read_csv('data/processed/processed_data.csv')

# Load metadata
with open('data/processed/metadata.json', 'r') as f:
    metadata = json.load(f)

# Print basic statistics
print("Dataset Overview:")
print("-" * 50)
print(f"Total number of employees: {len(processed_data)}")
print(f"Number of features: {len(metadata['feature_names'])}")
print(f"\nAttrition Rate: {processed_data['Attrition'].mean()*100:.2f}%")
print("\nFeature Types:")
print("Categorical:", ", ".join(metadata['categorical_columns']))
print("Numerical:", ", ".join(metadata['numerical_columns']))

# Calculate attrition by department
dept_attrition = processed_data.groupby('department_name')['Attrition'].agg(['mean', 'count'])
dept_attrition['mean'] = dept_attrition['mean'] * 100
dept_attrition = dept_attrition.sort_values('mean', ascending=False)

print("\nAttrition by Department:")
print("-" * 50)
for idx, row in dept_attrition.iterrows():
    print(f"{idx}: {row['mean']:.2f}% (Count: {row['count']})")

# Calculate average length of service for churned vs retained employees
service_stats = processed_data.groupby('Attrition')['length_of_service'].agg(['mean', 'std', 'count'])
print("\nLength of Service Statistics:")
print("-" * 50)
print("Retained Employees:")
print(f"Average: {service_stats.loc[0, 'mean']:.2f} years")
print(f"Std Dev: {service_stats.loc[0, 'std']:.2f} years")
print("\nChurned Employees:")
print(f"Average: {service_stats.loc[1, 'mean']:.2f} years")
print(f"Std Dev: {service_stats.loc[1, 'std']:.2f} years")

# Age distribution analysis
age_stats = processed_data.groupby('Attrition')['age'].agg(['mean', 'std'])
print("\nAge Statistics:")
print("-" * 50)
print("Retained Employees:")
print(f"Average Age: {age_stats.loc[0, 'mean']:.2f} years")
print(f"Std Dev: {age_stats.loc[0, 'std']:.2f} years")
print("\nChurned Employees:")
print(f"Average Age: {age_stats.loc[1, 'mean']:.2f} years")
print(f"Std Dev: {age_stats.loc[1, 'std']:.2f} years")
