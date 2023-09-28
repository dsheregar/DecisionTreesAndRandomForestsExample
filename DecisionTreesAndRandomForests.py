#Pre-requisites: run command pip install pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np 


data = pd.read_csv('HPI_master.csv')    #Imports csv file
data.dropna(inplace = True)             #Remove all rows with missing values

X = data[['hpi_flavor', 'level', 'yr', 'place_name']]
Y = data['index_nsa']

X_encoded = pd.get_dummies(X, columns=['hpi_flavor', 'level', 'place_name'], drop_first=True)

X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.25, random_state = 1)

dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(X_train, Y_train)

Y_pred_dt = dt_model.predict(X_test)

r2_dt = r2_score(Y_test, Y_pred_dt)
rmse_dt = np.sqrt(mean_absolute_error(Y_test, Y_pred_dt))
mae_dt = mean_absolute_error(Y_test, Y_pred_dt)

print("Decision Tree:")
print(f'R-squared: {r2_dt}')
print(f'Root Mean Squared Error: {rmse_dt}')
print(f'Mean Absolute Error: {mae_dt}')
#--------------------------------------------------------------------------------------------------------------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
rf_model.fit(X_train, Y_train)

Y_pred_rf = rf_model.predict(X_test)

r2_rf = r2_score(Y_test, Y_pred_rf)
rmse_rf = np.sqrt(mean_absolute_error(Y_test, Y_pred_rf))
mae_rf = mean_absolute_error(Y_test, Y_pred_rf)

print("\nRandom Forest:")
print(f'R-squared: {r2_rf}')
print(f'RMSE: {rmse_rf}')
print(f'MAE: {mae_rf}')