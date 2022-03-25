# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:53:43 2021

@author: ersuk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#veri yükleme

veriler = pd.read_csv('veriler.txt')

#veriler = pd.read_csv('eksikveriler.txt')
print(veriler)
#veri ön işleme

boy = veriler[['boy']]
print(boy)

#eksik veriler

#sci-kit learn

#from sklearn.impute import SimpleImputer

#imputer = SimpleImputer(missing_values= np.nan , strategy = 'mean')

#yas = veriler.iloc[:, 1:4].values
#print(yas)
#imputer = imputer.fit(yas[: ,1:4])
#yas[: ,1:4] = imputer.transform(yas[: ,1:4])
#print(yas)