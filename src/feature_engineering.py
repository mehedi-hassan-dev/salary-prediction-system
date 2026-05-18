# ------------import libraries------------------

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


# -----------------Load Cleane dataset--------------

df = pd.read_csv('data/processed/cleaned_salary_data.csv')
print(f'Dateset :\n{df.head()}')


# ------------------Dataset Validation---------------

print(f"\nBefore Encoding Dataset shape : {df.shape}")

print("\nBefore Encoding Dataset info :")
df.info()


# -------------------Gender Encoding (One Hot)------------------

df = pd.get_dummies(
    df,
    columns = ['Gender'],
    drop_first = True
)


# ---------------------Educational Level Encoding (Ordinal Encoding)-------------------
# Since there is a natural order in the Educational Level column(High School - Bachelor's - Master's - PhD)

education_order = [[
    "High School",
    "Bachelor's",
    "Master's",
    "PhD"
]]
encoder = OrdinalEncoder(categories = education_order)

df['Education Level'] = encoder.fit_transform(
    df[['Education Level']]
)


# ------------------Job Title Encoding(One Hot)---------------

df = pd.get_dummies(
    df,
    columns = ["Job Title"],
    drop_first = True
)


# ---------------Convert boolean columns-------------

# Select all boolean columns
bool_cols = df.select_dtypes(
    include='bool'
).columns

# Convert bool → int
df[bool_cols] = df[bool_cols].astype(int)


# -----------------Final Validation---------------

print(f'\nAfter Encoding Dataset Shape : {df.shape}')
print(f'\nAfter Encodin Dataset :\n{df.head()}')
print(f'\nAfter Encoding Dataset info :')
df.info()


# ----------------Save Dataset---------------------

df.to_csv('data/processed/featured_salary_data.csv', index = False)

print('\nFeature Engineering Completed')