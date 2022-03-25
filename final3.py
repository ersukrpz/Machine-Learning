
# Grid Search

from sklearn.model_selection import GridSearchCV
from preprocessing import *
from sklearn.metrics import classification_report
from classification import results
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Define the hyperparameter configuration space
model = results[results['Accuracy'] == results['Accuracy'].max()]['Model'].max()
rf_params = [
    {'C': [1, 50, 100, 1000]},
    {'gamma': [1e-3, 1e-4]},
    {'kernel': ('linear', 'rbf', 'sigmoid', 'poly')}]
# models['RF'] = RandomForestClassifier(random_state=0)
best_params = []
scores = []
for prmtr in rf_params:    
    print(prmtr)
    grid = GridSearchCV(SVC(), param_grid=prmtr, scoring='accuracy')
    grid.fit(rescaledX_train.tolist(), y_train.tolist())
    scores.append(grid.cv_results_)
    best_params.append(grid.best_params_)
    
    predictions = grid.predict(X_test_rescaled.tolist()) 
    # plt.scatter(rescaledX_train[:,9], y_train,c='b')
    # plt.scatter(rescaledX_train[:,9], predictions,c='r',alpha=0.5)
    # plt.show()
    print(classification_report(y_test, predictions.tolist())) 



params = [prm['params'] for prm in scores]
scores = [score['mean_test_score'] for score in scores]
c_values = [list(c.values())[0] for c in params[0]]
c_values = [str(x) for x in c_values]
gamma_values = [list(c.values())[0] for c in params[1]]
gamma_values = [str(x) for x in gamma_values]
kernel_values = [list(c.values())[0] for c in params[2]]
plt.bar(kernel_values, scores[2])
plt.show()
rf_params_com = {'C': [1, 50, 100, 1000], 'gamma': [1e-3, 1e-4],'kernel': ('linear', 'rbf', 'sigmoid', 'poly')}

grid = GridSearchCV(estimator=SVC(), param_grid=rf_params_com, scoring='accuracy', n_jobs=-1, verbose=42)
grid.fit(rescaledX_train, y_train)
d = grid.cv_results_

grid.fit(rescaledX_train, y_train)
d = grid.cv_results_
best_model = SVC()
best_model.fit(rescaledX_train, y_train)
y_pred = best_model.predict(X_test_rescaled)
accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# grid_scores = grid.cv_results_

# plot.grid_search(grid.cv_results_, change='n_estimators', kind='bar')

# print("Accuracy:"+ str(grid.best_score_))

#final = grid()

#final.plot(kind = "bar", figsize=(16,10))