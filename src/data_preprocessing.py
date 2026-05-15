
# -------import libraries--------

import pandas as pd
import numpy as np


# ---------Load Datasets-----------

df = pd.read_csv('data/raw/salary_data.csv')

print(df.head())


# ----------Basic Dataset information-----------

print(f'\nDataSet Shape :\n{df.shape}')

print(f'\nMissing Values :\n{df.isnull().sum()}')

print(f'\nDuplicate Rows :\n{df.duplicated().sum()}')


# ----------Missing Value Handling-----------------

# Remove rows with missing values
# (Dataset contains very few missing values)

print(f'\nBefore shape : {df.shape}')
df = df.dropna()
print(f'After shape : {df.shape}')

print(f'\nMissing Values :\n{df.isnull().sum()}')


# ------------Duplicate Value Removal--------------

print(f'\nDuplicate Rows : {df.duplicated().sum()}')
df = df.drop_duplicates()
print(f'After removal : {df.duplicated().sum()}')


# -------------Category Cleaning-------------------
# Education Level cleaning
# (Standardize inconsisten category name)

print(f"\nBefore Cleaning :\n{df['Education Level'].value_counts()}")

education_mapping = {
    "Bachelor's Degree": "Bachelor's",
    "Master's Degree": "Master's",
    "phD": "PhD"
}
df['Education Level'] = df['Education Level'].replace(education_mapping)

print(f"\nAfter Cleaning :\n{df['Education Level'].value_counts()}")


# ---------------Rare Category Handeling-----------------

print(f"\nBefore : {df['Job Title'].nunique()}")

# Categories with frequency below this threshold
RARE_THRESHOLD = 10

# Count Occurencess of each job title
job_counts = df['Job Title'].value_counts()

# Identify reare job titles
rare_jobs = job_counts[job_counts < RARE_THRESHOLD].index

# Replace rare categories with 'Others'
df['Job Title'] = df['Job Title'].replace(rare_jobs, 'Others')

# Validation
print(f"\nUpdate Job Title Distribution :\n{df['Job Title'].value_counts()}")

print(f"\nAfter : {df['Job Title'].nunique()}")


# -------------Remove duplicates created after category replacement-------------

df = df.drop_duplicates()


# ------------------Final Dataset Validation----------------

print(f'\nFinal Dataset Shape : {df.shape}')

print(f'\nFinal Missing Values : \n{df.isnull().sum()}')

print(f'\nFinal Duplicate Raws : {df.duplicated().sum()}')


# --------------------Save Processed Dataset---------------------

df.to_csv('data/processed/cleaned_salary_data.csv', index=False)
print(f'\nCleaned dataset save successfully')