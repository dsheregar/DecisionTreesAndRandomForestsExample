#Pre-requisites: run command pip install pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np


data = pd.read_csv('BostonHousing.csv')
data.dropna(inplace = True)

input_columns = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
output_columns = ['medv']
INPUT = data[input_columns]
OUTPUT = data[output_columns]

X_train, X_test, Y_train, Y_test = train_test_split(INPUT, OUTPUT, test_size=0.25, random_state = 1)

poly_features = PolynomialFeatures(degree = 3)      
X_train_poly = poly_features.fit_transform(X_train) 
X_test_poly = poly_features.transform(X_test)       

model = LinearRegression()          
model.fit(X_train_poly, Y_train)    

#user_input = float(input("Enter what year the home was built: "))   #Ask the user to input the X value
#user_input_poly = poly_features.transform(np.array([[user_input]])) #Transform the input into a 2D array (reason where there are two square brackets) and give it poly features
#user_input_standardized = scaler.transform(user_input_poly)
#predicted_output = model.predict(user_input_standardized)           #Use the model to predict the outcome
#print(f"Your home's HPI is: {predicted_output[0]}")                 #Output the predicted value based on the user's input

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error   #Import functions to calculate mean squared error (mse) and r2 score
Y_pred = model.predict(X_test_poly)                                             #Output a dataset Y_pred which is a list of answers based on the poly transformed X test set
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))                              #Calculate the mse of the dataset by comparing the predicted Y values using a transformed X set with the X's Y values
r2 = r2_score(Y_test, Y_pred)                                                   #Calculate the r2 score of the above two sets
mae = mean_absolute_error(Y_test, Y_pred)
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}") 

"""
data = pd.read_csv('HPI_master.csv')    #Imports csv file
data.dropna(inplace = True)             #Remove all rows with missing values

Year = data['yr'].values.reshape(-1, 1)                 #Reads the column with the title 'yr' and reshapes it to one column
PlaceName = data['place_name'].astype(str).values.reshape(-1, 1)
Index_NSA = data['index_nsa'].values                    #Reads the column with the title 'index_nsa'

encoder = OneHotEncoder(sparse_output=False)
print(data['yr'].values.shape)
X_categorical_encoded = encoder.fit_transform(PlaceName)
X_combined = np.concatenate((Year, X_categorical_encoded), axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Index_NSA, test_size=0.25, random_state = 1)
#X_train, X_test, Y_train, Y_test = train_test_split(Year, Index_NSA, test_size=0.25, random_state = 1)
#Splits the data into a train and test set, where the test set is 25% of the data
#random_state uses a fixed number as an id for a resultng shuffle of the dataset
#Use 1 for the rest of the project, or change it to literally any other number for a new shuffle

poly_features = PolynomialFeatures(degree = 3)      #Declaring the polynomial graph as two degrees
X_train_poly = poly_features.fit_transform(X_train) #fit_transform will give X_train 2nd degree polynomial features (eg: [1, 2, 3] becomes [ [1,1], [2,4], [3,9] ])
X_test_poly = poly_features.transform(X_test)       #From the same transformation used on X_train, apply the same changes to X_test
#The function transform uses the polynomial techniques learn from first applying fit_transform
#For being consistent, use fit_transform first on the train set, then use transform on the test set

scaler = StandardScaler()
X_train_poly = scaler.fit_transform(X_train_poly)
X_test_poly = scaler.transform(X_test_poly)

X_train_final = np.concatenate((X_train_poly, X_train[:, 1:]), axis = 1)
X_test_final = np.concatenate((X_test_poly, X_test[:, 1:]), axis = 1)

poly_features = PolynomialFeatures(degree = 3)
scaler = StandardScaler()

X_train_poly = np.empty((0, 0))
batch_size = 1000
for i in range(0, len(X_train), batch_size):
    X_train_batch = X_train[i:i + batch_size]
    X_train_batch_year_

model = LinearRegression()          #Make a Linear Regression model
model.fit(X_train_poly, Y_train)    #Use the transformed X_train set and the Y_train set to load the model

#user_input = float(input("Enter what year the home was built: "))   #Ask the user to input the X value
#user_input_poly = poly_features.transform(np.array([[user_input]])) #Transform the input into a 2D array (reason where there are two square brackets) and give it poly features
#user_input_standardized = scaler.transform(user_input_poly)
#predicted_output = model.predict(user_input_standardized)           #Use the model to predict the outcome
#print(f"Your home's HPI is: {predicted_output[0]}")                 #Output the predicted value based on the user's input

from sklearn.metrics import mean_squared_error, r2_score    #Import functions to calculate mean squared error (mse) and r2 score
Y_pred = model.predict(X_test_poly)                         #Output a dataset Y_pred which is a list of answers based on the poly transformed X test set
mse = mean_squared_error(Y_test, Y_pred)                    #Calculate the mse of the dataset by comparing the predicted Y values using a transformed X set with the X's Y values
r2 = r2_score(Y_test, Y_pred)                               #Calculate the r2 score of the above two sets
print(f"Mean Squared Error: {mse}")                         #Output the mse
print(f"R-squared: {r2}")                                   #Output the r2 score
"""