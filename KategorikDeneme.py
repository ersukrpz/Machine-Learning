# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:53:43 2021

@author: ersuk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#veri yükleme

#veriler = pd.read_csv('veriler.txt')

veriler = pd.read_csv('eksikveriler.txt')
print(veriler)

#veri ön işleme

#boy = veriler[['boy']]
#print(boy)

#eksik veriler

#sci-kit learn

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values= np.nan , strategy = 'mean')

yas = veriler.iloc[:, 1:4].values
print(yas)
imputer = imputer.fit(yas[: ,1:4])
yas[: ,1:4] = imputer.transform(yas[: ,1:4])
print(yas)
ulke = veriler.iloc[:, 0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[: ,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

print(list(range(22)))
sonuc = pd.DataFrame(data=ulke, index = range(22), columns= ['fr', 'tr', 'us'])
print(sonuc)

sonuc2 = pd.DataFrame(data= yas, index = range(22), columns=['boy','kilo','yaş'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet , index = range(22), columns = ['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc , sonuc2], axis=1)
print(s)

s2 = pd.concat([s,sonuc3], axis=1)
print(s2)