import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVR

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# read dataset
df = pd.read_csv(filepath_or_buffer='LasVegasTripAdvisorReviews-Dataset.csv', sep=';', header=0)

# get dataset's columns
cat_values = list(df.select_dtypes(include=['object']).columns.values)

# convert from object to int for our variables
le = LabelEncoder()


for i in range(0, len(cat_values)):
    df[cat_values[i]] = le.fit_transform(df[cat_values[i]])
    
X = df.drop('Score', axis=1).values
y = df['Score'].values

fig = plt.figure(1, figsize = (20,20))
ax = fig.gca()
df.drop('Score', axis=1).hist(bins=10,figsize=(16,16),grid=False, ax=ax)
plt.show()

# seperate test and train data from dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# set array for regression that we want to examine
reg = ['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'K-NN', 'Decision tree', 'Random forest', 'Support Vector Regressor']
reg_met = [LinearRegression(), Ridge(alpha=1.0), Lasso(alpha=0.1), ElasticNet(alpha=0.1),
           KNeighborsRegressor(n_neighbors=5),
           DecisionTreeRegressor(max_depth=3), RandomForestRegressor(n_estimators=20, random_state=0),
           SVR(C=1.0, epsilon=0.2)]



cv = KFold(n_splits=10, random_state=7, shuffle=True)
scores1 = pd.DataFrame({'Regression': [], 'R2 mean': [], 'MSE mean': [], 'R2 std': [], 'MSE std': []})
reg_scores1 = pd.DataFrame({'Regression': [], 'R2 train': [], 'MSE train': [], 'R2 test': [], 'MSE test': []})

j = 1
fig1 = plt.figure(2,figsize=(20, 20))
fig1.suptitle('Test predictions', fontsize=16)
fig2 = plt.figure(3,figsize=(20, 20))
fig2.suptitle('Train predictions', fontsize=16)
for i in range(len(reg)):
    regression = reg_met[i]
    model = regression.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results_cv = cross_validate(model, X, y, cv=cv, scoring=['r2', 'neg_mean_squared_error'])

    mse_mean = results_cv['test_neg_mean_squared_error'].mean()
    r2_mean= results_cv['test_r2'].mean()
    
    mse_std = results_cv['test_neg_mean_squared_error'].std()
    r2_std = results_cv['test_r2'].std() # standart deviation
    
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    scores1.loc[len(scores1)] = [reg[i], r2_mean, mse_mean, r2_std, mse_std]
    reg_scores1.loc[len(reg_scores1)] = [reg[i], r2_train, mse_train, r2_test, mse_test]
    
    curveaxis_test=np.zeros((100,X_test.shape[1]))
    for cx in range(X_test.shape[1]):
        curveaxis_test[:,cx]=np.linspace(np.min(X_test[:,cx]),np.max(X_test[:,cx]),100)
    curve_predictions_test = model.predict(curveaxis_test) 
    ax1 = fig1.add_subplot(5,3,j)
    
    ax1.set_title(reg[i])
    ax1.scatter(X_test[:,0], y_test,c='b')
    ax1.scatter(X_test[:,0], y_test_pred,c='r',alpha=0.5) 
    ax1.plot(curveaxis_test[:,0], curve_predictions_test,c='r')
    ax1.grid()
    
    curveaxis_train=np.zeros((100,X_train.shape[1]))
    for cx in range(X_train.shape[1]):
        curveaxis_train[:,cx]=np.linspace(np.min(X_train[:,cx]),np.max(X_train[:,cx]),100)
    
    curve_predictions_train = model.predict(curveaxis_train) 
    
    ax2 = fig2.add_subplot(5,3,j)
    ax2.set_title(reg[i]) 
    ax2.scatter(X_train[:,0], y_train,c='b') 
    ax2.scatter(X_train[:,0], y_train_pred,c='r',alpha=0.5) 
    ax2.plot(curveaxis_train[:,0], curve_predictions_train,c='r')
    ax2.grid()
    
    j+=1



fig3 = plt.figure(figsize = (20,20))
ax = fig3.gca()
scores1.hist(ax = ax)

scores1.plot('Regression',kind = 'bar', figsize=(16,10))





