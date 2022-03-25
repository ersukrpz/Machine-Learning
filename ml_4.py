from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
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

df.dtypes


# get dataset's columns
cat_values = list(df.select_dtypes(include=['object']).columns.values)

# convert from object to int for our variables
le = LabelEncoder() 

for i in range(0, len(cat_values)):
    df[cat_values[i]] = le.fit_transform(df[cat_values[i]], )
    df[cat_values[i]] = df[cat_values[i]].astype('int64')
    
df = df[df['Member years'] >= 0]


def vif_scores(df):
    VIF_Scores = pd.DataFrame()
    VIF_Scores["Independent Features"] = df.columns
    VIF_Scores["VIF Scores"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return VIF_Scores


df1 = df.drop(['Score'], axis=1)
vif_scores(df1)


df = df.drop(['Gym', 'Pool', 'Free internet'], axis=1)

# set array for regression that we want to examine
reg = ['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'K-NN', 'Decision tree', 'Random forest', 'Support Vector Regressor']
reg_met = [LinearRegression(), Ridge(alpha=1.0), Lasso(alpha=0.1), ElasticNet(alpha=0.1),
           KNeighborsRegressor(n_neighbors=5),
           DecisionTreeRegressor(max_depth=3), RandomForestRegressor(n_estimators=20, random_state=0),
           SVR(C=1.0, epsilon=0.2)]


X = df.drop('Score', axis=1).values
y = df['Score'].values

# seperate test and train data from dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# test verisinin çıkarımı
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

# transform the validation dataset
X_test_rescaled = scaler.transform(X_test)

scaled_features = StandardScaler().fit_transform(X)

variables = pd.DataFrame({'mse': [], 'Feature': []})

for x_ in scaled_features.T:
    X_train, X_test, y_train, y_test = train_test_split(x_.reshape(-1, 1), y, test_size=0.2, random_state=0)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    variables.loc[len(variables)] = [mse, x_]

variables.sort_values(by=['mse'], inplace=True)

variables = np.array(variables['Feature'][:int(len(variables) / 2) + 2].tolist()).T

# seperate test and train data from dataset
X_train, X_test, y_train, y_test = train_test_split(variables, y, test_size=0.2, random_state=42)

scores4 = pd.DataFrame({'Name': [], 'R2 mean': [], 'MSE mean': [], 'R2 std': [], 'MSE std': []})
cv = KFold(n_splits=10, random_state=7, shuffle=True)
plt.figure(figsize=(16, 16))
j = 1
for i in range(len(reg)):
    regression = reg_met[i]
    model = regression.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results_cv = cross_validate(model, X_test, y_test, cv=cv, scoring=['r2', 'neg_mean_squared_error'])

    mse_mean = results_cv['test_neg_mean_squared_error'].mean()
    r2_mean= results_cv['test_r2'].mean()
    
    mse_std = results_cv['test_neg_mean_squared_error'].std()
    r2_std = results_cv['test_r2'].std() # standart deviation

    scores4.loc[len(scores4)] = [reg[i], r2_mean, mse_mean, r2_std, mse_std]

    curveaxis=np.zeros((100,X_test.shape[1]))
    for cx in range(X_test.shape[1]):
        curveaxis[:,cx]=np.linspace(np.min(X_test[:,cx]),np.max(X_test[:,cx]),100) # linspace komutu başlangıç ve bitiş değerleri arasında belirtilen sayı kadar(100) parçalı değer oluşturur 
    curve_predictions = model.predict(curveaxis) 
    
    #tahmin ve rezidü çizimleri
    plt.subplot(5,3,j) # 5 satır 4 sütun çizim alanında i. çizim
    plt.title(reg[i]) # çizim başlığı
    plt.scatter(X_test[:,0], y_test,c='b') # test verisi 
    plt.scatter(X_test[:,0], y_test_pred,c='r',alpha=0.5) # test verisine karşılık prediction
    plt.plot(curveaxis[:,0], curve_predictions,c='r')# 0 sütunu değer atamaya karşılık tahminlerin eğri olarak çizilmesi
    plt.grid()
    
    j=j+1 # subplot indeksi
    j += 1
plt.show()

scores4.plot('Name',kind = 'bar')
