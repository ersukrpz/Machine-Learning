
import pandas as pd
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

df = pd.read_csv('weather.csv')

models=dict()
models['KNN'] = KNeighborsClassifier(n_neighbors=3)
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
	results = pd.DataFrame({'Classification':[], 'R2':[], 'Model':[]})
	for name, model in models.items():
		# evaluate the model
		results.loc[len(results)] = [name, evaluate_model(X, y, model), model]
	return results

results = evaluate_models(X, y, models)
results.plot('Classification',kind = 'bar', figsize=(16,10))