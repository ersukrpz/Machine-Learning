# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:28:57 2021

@author: Şehnaz Yıldırım
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



# data = pd.read_csv('drive/MyDrive/Colab Notebooks/LasVegasTripAdvisor/LasVegasTripAdvisorReviews-Dataset.csv', sep=';', header=0)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00397/LasVegasTripAdvisorReviews-Dataset.csv"
s=requests.get(url).content
data=pd.read_csv(io.StringIO(s.decode('utf-8')), sep=';', decimal=",")

data = data.replace("USA", 0).replace("UK", 1).replace("Canada", 2).replace("India", 3).replace("India ", 3).replace("Australia", 4).replace("New Zeland", 5).replace("Ireland", 6).replace("Egypt", 7).replace("Finland", 8).replace("Netherlands", 9).replace("Jordan", 10).replace("Kenya", 11).replace("Syria", 12).replace("Scotland", 13).replace("South Africa", 14).replace("Swiss", 15).replace("United Arab Emirates", 16).replace("Hungary", 17).replace("China", 18).replace("Greece", 19).replace("Mexico", 20).replace("Croatia", 21).replace("Germany", 22).replace("Malaysia", 23).replace("Thailand", 24).replace("Phillippines", 25).replace("Israel", 26).replace("Belgium", 27).replace("Puerto Rico", 28).replace("Switzerland", 29).replace("Norway", 30).replace("Spain", 31).replace("France", 32).replace("Singapore", 33).replace("Brazil", 34).replace("Costa Rica", 35).replace("Iran", 36).replace("Saudi Arabia", 37).replace("Honduras", 38).replace("Denmark", 39).replace("Taiwan", 40).replace("Hawaii", 41).replace("Kuwait", 42).replace("Czech Republic", 43).replace("Japan", 44).replace("Korea", 45).replace("Italy", 46).replace("NO", 0).replace("YES", 1).replace("Friends", 0).replace("Families", 1).replace("Couples", 2).replace("Solo", 3).replace("Business", 4).replace("Dec-Feb", 0).replace("Mar-May", 1).replace("Jun-Aug", 2).replace("Sep-Nov", 3).replace("Circus Circus Hotel & Casino Las Vegas", 0).replace("Excalibur Hotel & Casino", 1).replace("Monte Carlo Resort&Casino", 2).replace("Treasure Island- TI Hotel & Casino", 3).replace("Tropicana Las Vegas - A Double Tree by Hilton Hotel", 4).replace("Caesars Palace", 5).replace("The Cosmopolitan Las Vegas", 6).replace("The Palazzo Resort Hotel Casino", 7).replace("Wynn Las Vegas", 8).replace("Trump International Hotel Las Vegas", 9).replace("The Cromwell", 10).replace("Encore at wynn Las Vegas", 11).replace("Hilton Grand Vacations on the Boulevard", 12).replace("Marriott's Grand Chateau", 13).replace("Tuscany Las Vegas Suites & Casino", 14).replace("Hilton Grand Vacations at the Flamingo", 15).replace("Wyndham Grand Desert", 16).replace("The Venetian Las Vegas Hotel", 17).replace("Bellagio Las Vegas", 18).replace("Paris Las Vegas", 19).replace("The Westin las Vegas Hotel Casino & Spa", 20).replace("North America", 0).replace("Europe", 1).replace("Asia", 2).replace("Oceania", 3).replace("Africa", 4).replace("South America", 5).replace("January", 0).replace("February", 1).replace("March", 2).replace("April", 3).replace("May", 4).replace("June", 5).replace("July", 6).replace("August", 7).replace("September", 8).replace("October", 9).replace("November", 10).replace("December", 11).replace("Thursday", 0).replace("Friday", 1).replace("Saturday", 2).replace("Tuesday", 3).replace("Wednesday", 4).replace("Monday", 5).replace("Sunday", 6)

# giriş ve çıkış verşlerini ayırma
X = data.drop('Score', axis=1)
y = data.Score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xmeans=X_train.mean(axis=0)

#Standardization / Mean Removal / Variance Scaling
X_scaled = preprocessing.scale(X_train)
X_scaled.mean(axis=0)# mean of each coulmn=0.0
X_scaled.std(axis=0) # standard deviation of each coulmn=1.0

# scaler applied to train data
scaler = preprocessing.StandardScaler().fit(X_train)
scaler.transform(X_train)
plt.figure(figsize=(8,6))
plt.xlim(0, 1000)
plt.ylim(0, 1000)
plt.hist(X_train);

# scaler applied to test data
scaler.transform(X_test)
plt.figure(figsize=(8,6))
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.hist(X_test);

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MinMaxScaler
# Scale a data to the [0, 1] range-

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.transform(X_test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_test_maxabs = max_abs_scaler.transform(X_test)

#Normalization
X_normalized = preprocessing.normalize(X, norm='l2')

normalizer = preprocessing.Normalizer().fit(X) 
normalizer.transform(X)


# Binarization
binarizer = preprocessing.Binarizer().fit(X)
binarizer.transform(X)


# adjusting the binarization threshold
binarizer = preprocessing.Binarizer(threshold=-0.5)

#import dataset
#LA_data = pd.read_csv(r"C:\Users\damla\LasVegasTripAdvisorReviews-Dataset.csv", sep='delimiter', header=None)

#df = pd.DataFrame(data, columns=data.feature_names)
#df.head()

# defining the input and output
X = data[['Hotel name']].values # converts dataframe to numpy array
#y = data.target 

# scatter plot input vs output
plt.figure(figsize=(8,6))
plt.scatter(X, y);



# Before Scaling
plt.figure(figsize=(8,6))
plt.hist(X);
#plt.xlim(-40, 40);



## without pre-processing
# simple gradient descent regression algorithm
alpha = 0.0001
w_ = np.zeros(1 + X.shape[1])
cost_ = []
n_ = 100

for i in range(n_):
    y_pred = np.dot(X, w_[1:]) + w_[0]
    errors = (y - y_pred)
    
    w_[1:] += alpha * X.T.dot(errors)
    w_[0] += alpha * errors.sum()
    
    cost = (errors**2).sum() / 2.0
    cost_.append(cost)


# plot the error according to ephoc
plt.figure(figsize=(8,6))
plt.plot(range(1, n_ + 1), cost_);
plt.ylabel('SSE');
plt.xlabel('Epoch');  


## with pre-processing
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y.values.reshape(-1,1)).flatten()

#After Scaling
plt.figure(figsize=(8,6))
plt.hist(X_std);
plt.xlim(-4, 4);

# simple gradient descent regression algorithm
alpha = 0.0001
w_ = np.zeros(1 + X_std.shape[1])
cost_ = []
n_ = 100

for i in range(n_):
    y_pred = np.dot(X_std, w_[1:]) + w_[0]  # input variable changes as X_std
    errors = (y_std - y_pred)               # input variable changes as y_std
    
    w_[1:] += alpha * X_std.T.dot(errors)   # input variable changes as X_std 
    w_[0] += alpha * errors.sum()
    
    cost = (errors**2).sum() / 2.0
    cost_.append(cost)
plt.figure(figsize=(8,6))
plt.plot(range(1, n_ + 1), cost_);
plt.ylabel('SSE');
plt.xlabel('Epoch');
