import pandas as pd
import sklearn

# Building a Linear Regression Model

## Importing the data source
url = 'https://raw.githubusercontent.com/clebervisconti/datasets/main/delaney_solubility_with_descriptors.csv'

url_data = pd.read_table(url,sep='\t')

url_data.head(3)

df = pd.read_csv(url)

## creating the dataset
y = df['logS']

X = df.drop('logS', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

## aplicando regressão linear e treinando do modelo
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

print(y_lr_train_pred, y_lr_test_pred)

## verificando resultados
from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print('LR MSE (Train): ', lr_train_mse)
print('LR R2 (Train): ', lr_train_r2)
print('LR MSE (Test): ', lr_test_mse)
print('LR R2 (Test): ', lr_test_r2)

## resultado final
lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'TRaining R2', 'Test MSE', 'Test R2']

print(lr_results)


# Building a Randon Forest Model

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

## verificando resultados
from sklearn.metrics import mean_squared_error, r2_score
rf_train_mse = mean_squared_error(y_train, y_lr_train_pred)
rf_train_r2 = r2_score(y_train, y_lr_train_pred)

rf_test_mse = mean_squared_error(y_test, y_lr_test_pred)
rf_test_r2 = r2_score(y_test, y_lr_test_pred)

## resultado final
rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'TRaining R2', 'Test MSE', 'Test R2']

print(rf_results)

## Model Comparison

df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)

print(df_models)

# Data Visualization


import matplotlib.pyplot as plt
import numpy as np 

plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')
