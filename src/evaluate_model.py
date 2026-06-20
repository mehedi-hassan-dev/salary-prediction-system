import joblib
import pandas as pd 
import matplotlib.pyplot as plt

#-----------------Load Model-------------

model = joblib.load('models/best_model.pkl')

print(f'Model Type :\n{type(model)}')


#-------------Load Dataset-------------

df = pd.read_csv('data/processed/featured_salary_data.csv')
print(f'\nDataset Shape :\n{df.shape}')


#---------------Feature Importance-----------------

X = df.drop('Salary', axis=1)

importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})

importance_df = importance_df.sort_values(by='Importance', ascending = False)

print(f'\nTop 15 Important Features :\n{importance_df.head(15)}')


#--------------------Visualize Feature Importance-------------------

top_features = importance_df.head(15)
plt.figure(figsize=(10,6))

plt.barh(
    top_features['Feature'],
    top_features['Importance']
)
plt.title('Top 15 Important Features')
plt.tight_layout()
plt.show()


#----------------Prediction Testing--------------

sample = X.iloc[[0]]

prediction = model.predict(sample)
print(f'\nPredoction Testing :\n{prediction}')