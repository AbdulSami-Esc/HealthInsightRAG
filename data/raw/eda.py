import pandas as pd
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "healthcare_dataset.csv")
df = pd.read_csv(csv_path)

print("=== Test Results distribution ===")
print(df['Test Results'].value_counts())
print()

print("=== Admission Type distribution ===")
print(df['Admission Type'].value_counts())
print()

print("=== Age distribution ===")
print(df['Age'].describe())
print()

print("=== Cross-check: Abnormal results by Admission Type ===")
print(df[df['Test Results'] == 'Abnormal']['Admission Type'].value_counts())
print()

print("=== Average Billing by Admission Type ===")
print(df.groupby('Admission Type')['Billing Amount'].mean())