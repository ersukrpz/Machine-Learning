# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 01:20:45 2021

@author: ersuk
"""


import pandas as pd
import numpy as np
import io
import requests
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv(filepath_or_buffer='LasVegasTripAdvisorReviews-Dataset.csv', sep=';', decimal=",", header=0)

df = df.replace("USA", 0).replace("UK", 1).replace("Canada", 2).replace("India", 3).replace("India ", 3).replace("Australia", 4)\
    .replace("New Zeland", 5).replace("Ireland", 6).replace("Egypt", 7).replace("Finland", 8).replace("Netherlands", 9)\
    .replace("Jordan", 10).replace("Kenya", 11).replace("Syria", 12).replace("Scotland", 13).replace("South Africa", 14)\
    .replace("Swiss", 15).replace("United Arab Emirates", 16).replace("Hungary", 17).replace("China", 18).replace("Greece", 19)\
    .replace("Mexico", 20).replace("Croatia", 21).replace("Germany", 22).replace("Malaysia", 23).replace("Thailand", 24)\
    .replace("Phillippines", 25).replace("Israel", 26).replace("Belgium", 27).replace("Puerto Rico", 28).replace("Switzerland", 29)\
    .replace("Norway", 30).replace("Spain", 31).replace("France", 32).replace("Singapore", 33).replace("Brazil", 34)\
    .replace("Costa Rica", 35).replace("Iran", 36).replace("Saudi Arabia", 37).replace("Honduras", 38).replace("Denmark", 39)\
    .replace("Taiwan", 40).replace("Hawaii", 41).replace("Kuwait", 42).replace("Czech Republic", 43).replace("Japan", 44)\
    .replace("Korea", 45).replace("Italy", 46).replace("NO", 0).replace("YES", 1).replace("Friends", 0).replace("Families", 1)\
    .replace("Couples", 2).replace("Solo", 3).replace("Business", 4).replace("Dec-Feb", 0).replace("Mar-May", 1).replace("Jun-Aug", 2)\
    .replace("Sep-Nov", 3).replace("Circus Circus Hotel & Casino Las Vegas", 0).replace("Excalibur Hotel & Casino", 1)\
    .replace("Monte Carlo Resort&Casino", 2).replace("Treasure Island- TI Hotel & Casino", 3)\
    .replace("Tropicana Las Vegas - A Double Tree by Hilton Hotel", 4).replace("Caesars Palace", 5)\
    .replace("The Cosmopolitan Las Vegas", 6).replace("The Palazzo Resort Hotel Casino", 7).replace("Wynn Las Vegas", 8)\
    .replace("Trump International Hotel Las Vegas", 9).replace("The Cromwell", 10).replace("Encore at wynn Las Vegas", 11)\
    .replace("Hilton Grand Vacations on the Boulevard", 12).replace("Marriott's Grand Chateau", 13)\
    .replace("Tuscany Las Vegas Suites & Casino", 14).replace("Hilton Grand Vacations at the Flamingo", 15).\
    replace("Wyndham Grand Desert", 16).replace("The Venetian Las Vegas Hotel", 17).replace("Bellagio Las Vegas", 18)\
    .replace("Paris Las Vegas", 19).replace("The Westin las Vegas Hotel Casino & Spa", 20).replace("North America", 0)\
    .replace("Europe", 1).replace("Asia", 2).replace("Oceania", 3).replace("Africa", 4).replace("South America", 5)\
    .replace("January", 0).replace("February", 1).replace("March", 2).replace("April", 3).replace("May", 4).replace("June", 5)\
    .replace("July", 6).replace("August", 7).replace("September", 8).replace("October", 9).replace("November", 10)\
    .replace("December", 11).replace("Thursday", 0).replace("Friday", 1).replace("Saturday", 2).replace("Tuesday", 3)\
    .replace("Wednesday", 4).replace("Monday", 5).replace("Sunday", 6)
    
X = df.drop('Score', axis=1).values
y = df['Score'].values

raw_data = df
X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(X, y, test_size=0.2, random_state=42)

std = StandardScaler()
X_pre = std.fit_transform(X)
y_pre = std.fit_transform(y.reshape(-1,1)).flatten()
X_pre_train, X_pre_test, y_pre_train, y_pre_test = train_test_split(X_pre, y_pre, test_size=0.2, random_state=42)


results = pd.DataFrame({'Name': [], 'R2': [], 'MSE': [], 'Preprocessing R2': [], 'Preprocessin MSE': []})

reg = ['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'K-NN', 'Decision tree', 'Random forest', 'Support Vector Regressor']
reg_met = [LinearRegression(), Ridge(alpha=1.0), Lasso(alpha=0.1), ElasticNet(alpha=0.1),
           KNeighborsRegressor(n_neighbors=5),
           DecisionTreeRegressor(max_depth=3), RandomForestRegressor(n_estimators=20, random_state=0),
           SVR(kernel='linear')]


def apply_reg():
    for i in range(len(reg)):
        reg_raw = reg_met[i]
        reg_pre = reg_met[i]
        model_raw = reg_raw.fit(X_raw_train, y_raw_train)
        y_raw_pred = model_raw.predict(X_raw_test)
        mse_raw = mean_squared_error(y_raw_test, y_raw_pred)
        r2_raw = r2_score(y_raw_test, y_raw_pred)

        model_pre = reg_pre.fit(X_pre_train, y_pre_train)
        y_pre_pred = model_pre.predict(X_pre_test)
        mse_pre = mean_squared_error(y_pre_test, y_pre_pred)
        r2_pre = r2_score(y_pre_test, y_pre_pred)
        results.loc[len(results)] = [reg[i], r2_raw, mse_raw, r2_pre, mse_pre]

    return results

apply_reg()
    # Linear regression
lineer_regresyon_raw = LinearRegression()
model_raw = lineer_regresyon_raw.fit(X_raw_train, y_raw_train)
y_raw_pred = model_raw.predict(X_raw_test)
mse_raw = mean_squared_error(y_raw_test, y_raw_pred)
r2_raw = r2_score(y_raw_test, y_raw_pred)
lineer_regresyon_pre = LinearRegression()
model_pre = lineer_regresyon_pre.fit(X_pre_train, y_pre_train)
y_pre_pred = model_pre.predict(X_pre_test)
mse_pre = mean_squared_error(y_pre_test, y_pre_pred)
r2_pre = r2_score(y_pre_test, y_pre_pred)
results.loc[len(results)] = ['Linear', r2_raw, mse_raw, r2_pre, mse_pre]


