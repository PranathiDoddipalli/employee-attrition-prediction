import pandas as pd

# Load the dataset
df = pd.read_csv('c:/Users/prana/CascadeProjects/employee_attrition/data/raw/MFG10YearTerminationData.csv')

# Display basic information
print("Dataset Info:")
print(df.info())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst few rows:")
print(df.head())
