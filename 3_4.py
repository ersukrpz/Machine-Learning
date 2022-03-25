# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 23:05:31 2021

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
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from preprocessing import *
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('weather.csv')

models=dict()
models['KNN'] = KNeighborsClassifier(n_neighbors=7)
models['DT'] = DecisionTreeClassifier()
models['SVC'] = SVC()
models['GNB'] = GaussianNB()
models['LR'] = LogisticRegression(random_state=0)
models['RF'] = RandomForestClassifier(max_depth=2, random_state=0)
models['AB'] = AdaBoostClassifier(n_estimators=100, random_state=0)
models['MLP'] = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


cv = KFold(n_splits=10, random_state=7, shuffle=True)
def evaluate_model(X, y, model):
	# evaluate predictions
	result = cross_validate(model, X, y, cv=cv, scoring=['r2'])
	return result['test_r2'].mean() * 100.0


def evaluate_models(X, y, models):
	results = pd.DataFrame({'Classification':[], 'R2':[]})
	for name, model in models.items():
		# evaluate the model
		results.loc[len(results)] = [name, evaluate_model(X, y, model)]
	return results

#results = evaluate_models(X, y, models)
#results.plot('Classification',kind = 'bar', figsize=(16,10))

# Grid Search

#Random Forest
from sklearn.model_selection import GridSearchCV
# Define the hyperparameter configuration space
rf_params = {
    'n_estimators': [10, 20, 30],
    #'max_features': ['sqrt',0.5],
    'max_depth': [15,20,30,50],
    #'min_samples_leaf': [1,2,4,8],
    #"bootstrap":[True,False],
    "criterion":['gini','entropy']
}
models['RF'] = RandomForestClassifier(random_state=0)
grid = GridSearchCV(models['RF'], rf_params, cv=10, scoring='accuracy')
grid.fit(X, y)
print(grid.best_params_)
print("Accuracy:"+ str(grid.best_score_))

#grid_scores = grid.best_score_
#plot.grid_search(grid.best_score_, change='n_estimators', kind='bar')


import matplotlib.pyplot as plt
import numpy as np


def plotGraph(y_test,y_pred,regressorName):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.scatter(range(len(y_test)), y_test, color='blue')
    plt.scatter(range(len(y_pred)), y_pred, color='red')
    plt.title(regressorName)
    plt.show()
    return


y_test = range(10)
y_pred = np.random.randint(0, 10, 10)

plotGraph(y_test, y_pred, "test")


