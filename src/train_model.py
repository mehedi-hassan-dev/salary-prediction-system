# -------------import libraries-------------

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

# Evaluation Metrics
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
    )


#-------------Load Dataset--------------

df = pd.read_csv('data/processed/featured_salary_data.csv')

print(f'Dataset Shape : {df.shape}')


#--------------Feature & Target Split---------------

X = df.drop('Salary', axis=1)
y = df['Salary']

print(f'\nX shape is : {X.shape}')
print(f'\ny shape is : {y.shape}')


#--------------Train test split---------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'\nX_train Shape is : {X_train.shape}')
print(f'\nX_test Shape is : {X_test.shape}')

print(f'\ny_train Shape is : {y_train.shape}')
print(f'\ny_test Shape is : {y_test.shape}')


#---------------Models-----------------

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Graduent Boosting': GradientBoostingRegressor(random_state=42),
    'Extra Trees': ExtraTreesRegressor(random_state=42)
}


#---------------Model Training & Evaluation---------------------

results = []

for name,model in models.items():
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    mae =  mean_absolute_error(y_test, y_pred)
    
    rmse = root_mean_squared_error(y_test, y_pred)
    
    r2 = r2_score(y_test, y_pred)
    
    results.append([name, mae, rmse, r2])
    
    print(f'\n{name}')
    print(f'MAE : {mae:.2f}')
    print(f'RMSE : {rmse:.2f}')
    print(f'R2 : {r2:.4f}')
    
    
    print('Train Score :', model.score(X_train, y_train))
    print('Test Score :', model.score(X_test, y_test))
    
    
#---------------Model Comparison---------------------

results_df = pd.DataFrame(results, columns=['Model', 'MAE', 'RMSE', 'R2'])

results_df = results_df.sort_values(by='R2', ascending=False)

print(results_df)



#---------------------Best Model Selection--------------------

best_model_name = results_df.iloc[0]['Model']

print(f"\nBest Model : {best_model_name}")

    
#---------------Cross Validation----------------

rf = RandomForestRegressor(random_state=42)

cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')

print("\nCross Validation Scores :")
print(cv_scores)

print(f'\nAverage CV scores :\n{cv_scores.mean():.4f}')
print(f'\nCV Standard Deviation Score :\n{cv_scores.std():.4f}')


#----------------Hyperparameter Tuning--------------

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 15, 20, 30],
    'min_samples_split': [2, 4, 6, 10]
}

rf = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, 
n_iter = 20, cv = 5, scoring = 'r2', random_state = 42, n_jobs = -1)

random_search.fit(X_train, y_train)

print(f"\nBest Parameters :\n{random_search.best_params_}")
print(f"\nBest CV Score :\n{random_search.best_score_:.4f}")



#best_rf: RandomForestRegressor = random_search.best_estimator_
best_rf = random_search.best_estimator_

best_rf = RandomForestRegressor(**random_search.best_params_)
best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = root_mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

train_score = best_rf.score(X_train, y_train)

test_score = best_rf.score(X_test, y_test)

print('\nTuned Random Forest')

print(f'\nMAE : {mae:.2f}')
print(f'\nRMSE : {rmse:.2f}')
print(f'\nR2 : {r2:.4f}')

print(f'Train Score : {train_score:.4f}')
print(f'Test Score : {test_score:.4f}')


#--------------Save Best Model--------------

joblib.dump(best_rf,'models/best_model.pkl')