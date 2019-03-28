
# coding: utf-8

# In[64]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

train = pd.read_csv('E:\\train (1).csv')
test = pd.read_csv('E:\\test (1).csv')

train.drop('ID',axis=1,inplace=True)
test.drop('ID',axis=1,inplace=True)

column_names = test.columns

Q1 = train.quantile(0.25)
Q3 = train.quantile(0.75)
IQR = Q3 - Q1
#print(IQR)

train_dum = train.copy()
test_dum = test.copy()

a = ((train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR)))

for i,j in zip(np.arange(len(train.median())),train.median()):
    (train_dum.ix[a.iloc[:,i],i]) = j

    
    
g1 = train.median().drop('medv')
 
    
min2 = (Q1 - 1.5 * IQR).drop('medv')
max2 = (Q3 + 1.5 * IQR).drop('medv')
   
for i,j in zip(np.arange(len(g1)),g1):
    
    test_dum.ix[(test_dum.iloc[:,i]>max2[i]) | (test_dum.iloc[:,i]<min2[i]),i] =j

train = train_dum.copy()
test = test_dum.copy()

train['income_cat'] = np.floor(train['lstat']/4.2)
train['income_cat'].where(train['income_cat']<5,5,inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(train, train["income_cat"]):
    strat_train_set = train.loc[train_index]
    strat_test_set = train.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

train = strat_train_set.copy()


X_train = train.drop('medv',axis=1).copy()
y_train = train['medv'].copy()
X_test = strat_test_set.drop('medv',axis=1).copy()
y_test =strat_test_set['medv'].copy()


#scaling the terms
from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler()
scale_y = StandardScaler()
X_train = scale_x.fit_transform(X_train)
X_test = scale_x.transform(X_test)
y_train = scale_y.fit_transform(y_train.values.reshape(-1,1))
y_test = scale_y.transform(y_test.values.reshape(-1,1))


test = scale_x.transform(test)

y_train = y_train.ravel()
y_test = y_test.ravel()

X_train = pd.DataFrame(X_train,columns=column_names)
X_test = pd.DataFrame(X_test,columns=column_names)

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
pred_1 = lin_reg.predict(X_train)
error_1 = np.sqrt(mean_squared_error(y_train,pred_1))
#error_1

rid_reg = Ridge(alpha=1,solver='cholesky')
rid_reg.fit(X_train,y_train)
pred_2 = rid_reg.predict(X_train)
error_2 = np.sqrt(mean_squared_error(y_train,pred_2))
#error_2

lass_reg = Lasso(alpha=0.1)
lass_reg.fit(X_train,y_train)
pred_3 = lass_reg.predict(X_train)
error_3 = np.sqrt(mean_squared_error(y_train,pred_3))
#error_3

sgd_reg = SGDRegressor(n_iter=50,penalty=None,eta0=0.1)
sgd_reg.fit(X_train,y_train)
pred_4 = sgd_reg.predict(X_train)
error_4 = np.sqrt(mean_squared_error(y_train,pred_4))
#error_4

knr_reg = KNeighborsRegressor(n_neighbors=3)
knr_reg.fit(X_train,y_train)
pred_5 = knr_reg.predict(X_train)
error_5 = np.sqrt(mean_squared_error(y_train,pred_5))
#error_5

svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train,y_train)
pred_6 = svr_reg.predict(X_train)
error_6 = np.sqrt(mean_squared_error(y_train,pred_6))
#error_6

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train,y_train)
pred_7 = tree_reg.predict(X_train)
error_7 = np.sqrt(mean_squared_error(y_train,pred_7))
#error_7

forest_reg = RandomForestRegressor(random_state=42,n_jobs=-1,oob_score=True)
forest_reg.fit(X_train,y_train)
pred_8 = forest_reg.predict(X_train)
error_8 = np.sqrt(mean_squared_error(y_train,pred_8))
#error_8

elasic_reg = ElasticNet(alpha=0.1,l1_ratio=0.5)
elasic_reg.fit(X_train,y_train)
pred_9 = elasic_reg.predict(X_train)
error_9 = np.sqrt(mean_squared_error(y_train,pred_9))
#error_9

svr_reg_2 = SVR(kernel='sigmoid')
svr_reg_2.fit(X_train,y_train)
pred_10 = svr_reg_2.predict(X_train)
error_10 = np.sqrt(mean_squared_error(y_train,pred_10))
#error_10

svr_reg_3 = SVR(kernel='poly',degree=2)
svr_reg_3.fit(X_train,y_train)
pred_11 = svr_reg_3.predict(X_train)
error_11 = np.sqrt(mean_squared_error(y_train,pred_11))
#error_11

svr_reg_4 = SVR(kernel='poly',degree=3)
svr_reg_4.fit(X_train,y_train)
pred_12 = svr_reg_3.predict(X_train)
error_12 = np.sqrt(mean_squared_error(y_train,pred_12))
#error_12



results = pd.DataFrame([error_1,error_2,error_3,error_4,error_5,error_6,error_7,error_8,error_9,error_10,error_11,error_12],
                       index=['LinearRegression','Ridge','Lasso','SGDRegressor','KNeighborsRegressor','SVR','DecisionTreeRegressor',
                              'RandomForestRegressor','ElasticNet','sigmoid','poly_degree1','poly_degree2'],
                       columns=['mean_squared_error'])
results = results['mean_squared_error'].sort_values()
results = round(results,4)
results


# In[65]:


"""def display_scores(p):
    print(p.mean())
    print(p.std())"""

from sklearn.model_selection import cross_val_score

scores_1 = cross_val_score(tree_reg,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
tree_scores = np.sqrt(-scores_1)
#display_scores(tree_scores)

scores_2 = cross_val_score(forest_reg,X_train,y_train,scoring='neg_mean_squared_error',cv=10,n_jobs=-1)
forest_scores = np.sqrt(-scores_2)
#display_scores(forest_scores)

scores_3 = cross_val_score(knr_reg,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
knr_scores = np.sqrt(-scores_3)
#display_scores(knr_scores)

scores_4 = cross_val_score(svr_reg,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
svr_scores = np.sqrt(-scores_4)
#display_scores(svr_scores)

scores_5 = cross_val_score(lin_reg,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
lin_scores = np.sqrt(-scores_5)
#display_scores(svr_scores)

a = []
b = []

for i in [tree_scores,forest_scores,knr_scores,svr_scores,lin_scores]:
    a.append(i.mean())
    b.append(i.std())

a = np.array(a)
b = np.array(b)

df = pd.DataFrame([a,b,results[:5].values]).T
df.columns = ['mean_of_cross_val','std_of_cross_val','model_without_cross_val']
df.index = results[:5].index

print(df)


# In[66]:


#shortlist random knr linear
import statsmodels.formula.api as sm
X = np.append(np.ones((X_train.shape[0],1)),X_train,axis=1)
np.arange(X.shape[1])


# In[67]:


X_opt = X[:,[1,6, 7,  8,  10, 11,  13]].copy()
regressor_OLS = sm.OLS(endog= y_train,exog=X_opt).fit()
regressor_OLS.summary()


# In[68]:


np.array([1,6, 7,  8,  10, 11,  13])-1


# In[69]:


lin_reg_1 = LinearRegression()
lin_reg_1.fit(X_train.iloc[:,[ 0,  5,  6,  7,  9, 10, 12]],y_train)
pred_11 = lin_reg_1.predict(X_train.iloc[:,[ 0,  5,  6,  7,  9, 10, 12]])
error_11 = np.sqrt(mean_squared_error(y_train,pred_11))


forest_reg_11 = RandomForestRegressor(random_state=42,n_jobs=-1,oob_score=True)
forest_reg_11.fit(X_train.iloc[:,[ 0,  5,  6,  7,  9, 10, 12]],y_train)
pred_88 = forest_reg_11.predict(X_train.iloc[:,[ 0,  5,  6,  7,  9, 10, 12]])
error_88 = np.sqrt(mean_squared_error(y_train,pred_88))




knr_reg_22 = SVR(kernel='rbf')
knr_reg_22.fit(X_train.iloc[:,[ 0,  5,  6,  7,  9, 10, 12]],y_train)
pred_101 = knr_reg_22.predict(X_train.iloc[:,[ 0,  5,  6,  7,  9, 10, 12]])
error_101 = np.sqrt(mean_squared_error(y_train,pred_101))


results = pd.DataFrame([error_11,error_101,error_88],
                       index=['LinearRegression','SVR','RandomForestRegressor'],
                       columns=['mean_squared_error'])
results = results['mean_squared_error'].sort_values()
results = round(results,4)
results


# In[70]:



scores_2 = cross_val_score(forest_reg_11,X_train.iloc[:,[ 0,  5,  6,  7,  9, 10, 12]],y_train,scoring='neg_mean_squared_error',cv=10,n_jobs=-1)
forest_scores = np.sqrt(-scores_2)
#display_scores(forest_scores)

scores_3 = cross_val_score(knr_reg_22,X_train.iloc[:,[ 0,  5,  6,  7,  9, 10, 12]],y_train,scoring='neg_mean_squared_error',cv=10)
knr_scores = np.sqrt(-scores_3)
#display_scores(knr_scores)

scores_5 = cross_val_score(lin_reg_1,X_train.iloc[:,[ 0,  5,  6,  7,  9, 10, 12]],y_train,scoring='neg_mean_squared_error',cv=10)
lin_scores = np.sqrt(-scores_5)
#display_scores(svr_scores)

a = []
b = []

for i in [knr_scores,forest_scores,lin_scores]:
    a.append(i.mean())
    b.append(i.std())

a = np.array(a)
b = np.array(b)

df = pd.DataFrame([a,b]).T
df.columns = ['mean_of_cross_val','std_of_cross_val']
df.index = ['svr','forest','linear']

print(df)


# In[ ]:


#since in above thing svr has the lowest mean of cross val go for it


# In[77]:


#no point in going for random forest since the score is high for that in cross val score
#instead go for svr and then linear


# In[80]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.2,0.5,1,1.4],
             'gamma':[0.1,0.3,0.2,0.4,0.6,0.8,1],
             'tol':[1e-3,10e-3],
             'max_iter':[5,10,-1]}
forest_reg_2 = SVR(kernel='rbf')
grid_search = GridSearchCV(forest_reg_2,param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
grid_search.fit(X_train.iloc[:,[ 0,5,  6,  7,   9, 10, 12]],y_train)


# In[81]:


grid_search.best_params_


# In[82]:


cvres = grid_search.cv_results_
np.sqrt(-cvres['mean_test_score']).min()


# In[85]:


#since the above thing increased so go with the normal thing


# In[86]:


y_pred = knr_reg_22.predict(X_test.iloc[:,[ 0,  5,  6,  7,  9, 10, 12]])
error_11 = np.sqrt(mean_squared_error(y_test,y_pred))
error_11


# In[87]:


scale_y.inverse_transform(y_pred)


# In[88]:


scale_y.inverse_transform(y_test)


# In[91]:


scale_y.inverse_transform(knr_reg_22.predict(test[:,[ 0,  5,  6,  7,  9, 10, 12]]))


# In[94]:


#visualizing the results
from sklearn.decomposition import PCA


# In[99]:



pca = PCA(n_components=1)
principalComponents = pca.fit_transform(X_train.iloc[:,[ 0,5,  6,  7,   9, 10, 12]])


# In[100]:


pca.explained_variance_ratio_


# In[103]:


plt.scatter(principalComponents,scale_y.inverse_transform(y_train),c='r')
plt.scatter(principalComponents,scale_y.inverse_transform(knr_reg_22.predict(X_train.iloc[:,[ 0,  5,  6,  7,  9, 10, 12]])))


# In[122]:


#most dependent feature is lstat but it has negative dependency
#second most dependent feature is tax it has negative dependency
#third most dependent feature is rm it has positive dependency
#fourth most dependent feature is age it has negative dependency

