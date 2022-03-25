# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 01:30:40 2021

@author: aktas
"""

import pandas as pd
import numpy as np
import io
import requests
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00397/LasVegasTripAdvisorReviews-Dataset.csv"
s=requests.get(url).content
data=pd.read_csv(io.StringIO(s.decode('utf-8')), sep=';', decimal=",")

data = data.replace("USA", 0).replace("UK", 1).replace("Canada", 2).replace("India", 3).replace("India ", 3).replace("Australia", 4).replace("New Zeland", 5).replace("Ireland", 6).replace("Egypt", 7).replace("Finland", 8).replace("Netherlands", 9).replace("Jordan", 10).replace("Kenya", 11).replace("Syria", 12).replace("Scotland", 13).replace("South Africa", 14).replace("Swiss", 15).replace("United Arab Emirates", 16).replace("Hungary", 17).replace("China", 18).replace("Greece", 19).replace("Mexico", 20).replace("Croatia", 21).replace("Germany", 22).replace("Malaysia", 23).replace("Thailand", 24).replace("Phillippines", 25).replace("Israel", 26).replace("Belgium", 27).replace("Puerto Rico", 28).replace("Switzerland", 29).replace("Norway", 30).replace("Spain", 31).replace("France", 32).replace("Singapore", 33).replace("Brazil", 34).replace("Costa Rica", 35).replace("Iran", 36).replace("Saudi Arabia", 37).replace("Honduras", 38).replace("Denmark", 39).replace("Taiwan", 40).replace("Hawaii", 41).replace("Kuwait", 42).replace("Czech Republic", 43).replace("Japan", 44).replace("Korea", 45).replace("Italy", 46).replace("NO", 0).replace("YES", 1).replace("Friends", 0).replace("Families", 1).replace("Couples", 2).replace("Solo", 3).replace("Business", 4).replace("Dec-Feb", 0).replace("Mar-May", 1).replace("Jun-Aug", 2).replace("Sep-Nov", 3).replace("Circus Circus Hotel & Casino Las Vegas", 0).replace("Excalibur Hotel & Casino", 1).replace("Monte Carlo Resort&Casino", 2).replace("Treasure Island- TI Hotel & Casino", 3).replace("Tropicana Las Vegas - A Double Tree by Hilton Hotel", 4).replace("Caesars Palace", 5).replace("The Cosmopolitan Las Vegas", 6).replace("The Palazzo Resort Hotel Casino", 7).replace("Wynn Las Vegas", 8).replace("Trump International Hotel Las Vegas", 9).replace("The Cromwell", 10).replace("Encore at wynn Las Vegas", 11).replace("Hilton Grand Vacations on the Boulevard", 12).replace("Marriott's Grand Chateau", 13).replace("Tuscany Las Vegas Suites & Casino", 14).replace("Hilton Grand Vacations at the Flamingo", 15).replace("Wyndham Grand Desert", 16).replace("The Venetian Las Vegas Hotel", 17).replace("Bellagio Las Vegas", 18).replace("Paris Las Vegas", 19).replace("The Westin las Vegas Hotel Casino & Spa", 20).replace("North America", 0).replace("Europe", 1).replace("Asia", 2).replace("Oceania", 3).replace("Africa", 4).replace("South America", 5).replace("January", 0).replace("February", 1).replace("March", 2).replace("April", 3).replace("May", 4).replace("June", 5).replace("July", 6).replace("August", 7).replace("September", 8).replace("October", 9).replace("November", 10).replace("December", 11).replace("Thursday", 0).replace("Friday", 1).replace("Saturday", 2).replace("Tuesday", 3).replace("Wednesday", 4).replace("Monday", 5).replace("Sunday", 6)

#Performance Evaluation of Regression Model
X = data.drop('Score', axis=1).values
y = data['Score'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Residual Analysis
plt.figure(figsize=(12,8))
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=10, lw=2, color='k')
plt.xlim([-5, 10])
plt.ylim([-5,5])
plt.show()

# Mean Squared Error (MSE)
mean_squared_error(y_train, y_train_pred)
mean_squared_error(y_test, y_test_pred)

# Coefficient of Determination,  R2
r2_score(y_train, y_train_pred)
r2_score(y_test, y_test_pred)
