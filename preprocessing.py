# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 00:18:02 2021

@author: aktas
"""

import pandas as pd
from datetime import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

knn = KNeighborsClassifier(n_neighbors=7)

df = pd.read_csv('weather.csv')

# condition = ['Yağmur', 'Açık', 'Parçalı bulutlu', 'Gökgürültüsü', 'Az bulutlu', 'Dondurucu sis', 'Kar ', 'Bulutlu', 'Yoğun kar yağışı']

df.dtypes
len(df['Durum'].unique())
# df['Date'] = list(map(dateutil.parser.parse, df['Date']))

df['Date'] = [dt.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp() for date in df['Date']]
# df.to_csv(r'weather-timestamp.csv')
# df['Date'].apply(lambda x: x.toordinal())[0:10]

# get dataset's columns
cat_values = list(df.select_dtypes(include=['object']).columns.values)


# convert from object to int for our variables
le = LabelEncoder() 

for i in range(0, len(cat_values)):
    df[cat_values[i]] = le.fit_transform(df[cat_values[i]], )
    df[cat_values[i]] = df[cat_values[i]].astype('int64')

X = df.drop('Durum', axis=1).values
y = df['Durum'].values

v = X.var()

sel = VarianceThreshold(threshold=(2.9))
m = sel.fit_transform(X)

features = pd.DataFrame(data = m, columns= df.drop(['Enlem','Durum'], axis=1).columns)

rescaledX = StandardScaler().fit_transform(features)
X = pd.DataFrame(data = rescaledX, columns= features.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# test verisinin çıkarımı
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

# transform the validation dataset
X_test_rescaled = scaler.transform(X_test)

# deneme amaçlı bu kısım
knn.fit(X_train, y_train)
y_test_pred = knn.predict(X_test)
y_train_pred = knn.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)