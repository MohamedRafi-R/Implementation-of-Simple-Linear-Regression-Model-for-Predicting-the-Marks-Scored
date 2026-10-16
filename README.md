# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the student scores dataset using Pandas.

2.Separate the input feature (Hours) and target value (Scores), then split the dataset into training and testing sets.

3.Create and train the Simple Linear Regression model using the training data.

4.Predict marks for the test and new input data, plot the regression line, and evaluate the model using MAE, MSE, and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Mohamed Rafi R
RegisterNumber:  212224040195

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('exp_2_dataset_student_scores.csv')   # CSV should have two columns, e.g. "Hours","Scores"
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")
# Assume CSV columns: Hours (feature) and Scores (target)
X = df.iloc[:, :-1].values   # all rows, all columns except last -> shape (n_samples, 1)
Y = df.iloc[:, -1].values    # all rows, last column -> shape (n_samples,)
print("X (features):", X.flatten())
print("Y (targets):", Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))
regressor = LinearRegression()
regressor.fit(X_train, Y_train)   # fit on training data
Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)
plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
new_hours = np.array([[2.5], [8.0]])   # shape must be (n_samples, 1)
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

<img width="1113" height="452" alt="image" src="https://github.com/user-attachments/assets/2f646753-ad5c-4c59-8825-1bb3cf28b2f8" />

<img width="882" height="267" alt="image" src="https://github.com/user-attachments/assets/f02b175d-acad-4b44-9459-731b6e927c3e" />

<img width="858" height="145" alt="image" src="https://github.com/user-attachments/assets/651fc115-882d-483d-b831-72fa6f5c0249" />

<img width="771" height="407" alt="image" src="https://github.com/user-attachments/assets/fa62defb-bd56-4f68-b141-d75e28100a50" />

<img width="690" height="507" alt="image" src="https://github.com/user-attachments/assets/99fc24c0-4072-4c6a-9d83-c7a7dc3e8da9" />

<img width="687" height="510" alt="image" src="https://github.com/user-attachments/assets/7a90f5c4-ab72-45bd-96ed-eb6d69223e74" />

<img width="830" height="493" alt="image" src="https://github.com/user-attachments/assets/a6835b6f-b041-4a4c-8164-51cb3c69efb1" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
