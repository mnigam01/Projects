
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_auc_score,precision_score,recall_score,f1_score


# In[3]:


from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()


# In[4]:


data.keys()


# In[5]:


X = pd.DataFrame(data['data'],columns=data['feature_names'])
y = data['target']


# In[8]:


type(X) , type(y)


# In[9]:


X.info()


# In[11]:


X.dtypes


# In[15]:


X.isnull().sum()   #to check for the missing values


# In[16]:


#no missing values and no categorical things


# In[17]:


#removing outliers 
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
#print(IQR)

train_dum = X.copy()
#test_dum = X.copy()

a = ((X < (Q1 - 1.5 * IQR)) |(X > (Q3 + 1.5 * IQR)))

for i,j in zip(np.arange(len(X.median())),X.median()):
    (train_dum.ix[a.iloc[:,i],i]) = j


# In[18]:


train_dum.boxplot()


# In[19]:


X.boxplot()


# In[20]:


X_train,X_test,y_train,y_test = train_test_split(train_dum,y,test_size=0.2,stratify=y,random_state=42)


# In[21]:


column_names = X_train.columns
#scaling the terms
from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler()
scale_y = StandardScaler()
X_train = scale_x.fit_transform(X_train)
X_test = scale_x.transform(X_test)


X_train = pd.DataFrame(X_train,columns=column_names)
X_test = pd.DataFrame(X_test,columns=column_names)


# In[22]:


model1 = SGDClassifier(max_iter=5, tol=None)
model1.fit(X_train, y_train)
acc_1 = round(model1.score(X_train, y_train) * 100, 2)


model2 = RandomForestClassifier(n_estimators=100)
model2.fit(X_train, y_train)
acc_2 = round(model2.score(X_train, y_train) * 100, 2)


model3 = LogisticRegression()
model3.fit(X_train, y_train)
acc_3 = round(model3.score(X_train, y_train) * 100, 2)


model4 = KNeighborsClassifier(n_neighbors = 3)
model4.fit(X_train, y_train) 
acc_4 = round(model4.score(X_train, y_train) * 100, 2)


model5 = GaussianNB() 
model5.fit(X_train, y_train)
acc_5 = round(model5.score(X_train, y_train) * 100, 2)


model6 = Perceptron(max_iter=5)
model6.fit(X_train, y_train)
acc_6 = round(model6.score(X_train, y_train) * 100, 2)


model7 = SVC(kernel='rbf',random_state=42)
model7.fit(X_train,y_train)
acc_7 = round(model7.score(X_train,y_train)*100,2)


model8 = SVC(kernel='sigmoid',random_state=42)
model8.fit(X_train,y_train)
acc_8 = round(model8.score(X_train,y_train)*100,2)


model9 = SVC(kernel='poly',degree=2,random_state=42)
model9.fit(X_train,y_train)
acc_9 =round(model9.score(X_train,y_train)*100,2)


model10 = SVC(kernel='poly',degree=3,random_state=42)
model10.fit(X_train,y_train)
acc_10 =round(model10.score(X_train,y_train)*100,2)


model11 = SVC(kernel='poly',degree=4,random_state=42)
model11.fit(X_train,y_train)
acc_11 =round(model11.score(X_train,y_train)*100,2)


model12 = DecisionTreeClassifier()
model12.fit(X_train, y_train) 
acc_12 = round(model12.score(X_train, y_train) * 100, 2)

results = pd.Series( [acc_1, acc_2,acc_3, acc_4,acc_5,acc_6,acc_7,acc_8,acc_9,acc_10,acc_11,acc_12],
                    index =  ['Stochastic Gradient Decent','Random Forest','Logistic Regression','KNC', 
                     'Naive Bayes', 'Perceptron','svcrbf','svcsigmoid','svcpoly2','svcpoly3','svcpoly4',
                              'Decision Tree'])
results = results.sort_values(ascending=False)
results


# In[24]:


#select the first five classifiers with highest scores decision random svcwithrbf logisticresg knn svcsigmoid 


# In[29]:



rf = DecisionTreeClassifier()
scores_1 = cross_val_score(rf,  X_train,y_train, cv=10, scoring = "accuracy")


rf = RandomForestClassifier(n_estimators=100)
scores_2 = cross_val_score(rf, X_train,y_train, cv=10, scoring = "accuracy")



rf = SVC(kernel='rbf',random_state=42)
scores_3 = cross_val_score(rf,  X_train,y_train, cv=10, scoring = "accuracy")


rf = KNeighborsClassifier(n_neighbors = 3)
scores_4 = cross_val_score(rf,  X_train,y_train, cv=10, scoring = "accuracy")


rf = LogisticRegression()
scores_5 = cross_val_score(rf,  X_train,y_train, cv=10, scoring = "accuracy")


rf = SVC(kernel='sigmoid',random_state=42)
scores_6 = cross_val_score(rf,  X_train,y_train, cv=10, scoring = "accuracy")


a = []
b = []

for i in [scores_1,scores_2,scores_3,scores_4,scores_5,scores_6]:
    a.append(i.mean())
    b.append(i.std())

a = np.array(a)
b = np.array(b)

df = pd.DataFrame([a,b,results[:6].values]).T
df.columns = ['mean_of_cross_val','std_of_cross_val','model_without_cross_val']
df.index = results[:6].index

print(df)


# In[21]:


#if i were to use full features then i'll use svc with sigmoid kernel


# In[22]:


X_train_dummy = X_train.copy()
X_test_dummy = X_test.copy()


# In[24]:


#visualizing the results
from sklearn.decomposition import PCA
pca = PCA(n_components=19)
X_train= pca.fit_transform(X_train)
np.round(pca.explained_variance_ratio_,2)


# In[25]:


X_test = pca.transform(X_test)


# In[26]:


sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, y_train)
acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train) 
acc_knn = round(knn.score(X_train, y_train) * 100, 2)


gaussian = GaussianNB() 
gaussian.fit(X_train, y_train)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)


perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, y_train)
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)


svc_rbf = SVC(kernel='rbf',random_state=42)
svc_rbf.fit(X_train,y_train)
acc_1 = round(svc_rbf.score(X_train,y_train)*100,2)


svc_sigmoid = SVC(kernel='sigmoid',random_state=42)
svc_sigmoid.fit(X_train,y_train)
acc_2 = round(svc_sigmoid.score(X_train,y_train)*100,2)


svc_linear = SVC(kernel='poly',degree=2,random_state=42)
svc_linear.fit(X_train,y_train)
acc_3=round(svc_linear.score(X_train,y_train)*100,2)


svc_linear = SVC(kernel='poly',degree=3,random_state=42)
svc_linear.fit(X_train,y_train)
acc_4=round(svc_linear.score(X_train,y_train)*100,2)


svc_linear = SVC(kernel='poly',degree=4,random_state=42)
svc_linear.fit(X_train,y_train)
acc_5=round(svc_linear.score(X_train,y_train)*100,2)


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train) 
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

results = pd.Series( [ acc_knn, acc_log,acc_random_forest, acc_gaussian,acc_perceptron,acc_sgd,
                      acc_decision_tree,acc_1,acc_2,acc_3,acc_4,acc_5],
                    index =  ['KNN', 'Logistic Regression',
                    'Random Forest', 'Naive Bayes', 'Perceptron','Stochastic Gradient Decent',
                    'Decision Tree','svcrbf','svcsigmoid','svcpoly2','svcpoly3','svcpoly4'])
results = results.sort_values(ascending=False)
results


# In[ ]:


#svc with 3 degree polynomial is doing a good job now 


# In[27]:



rf = RandomForestClassifier(n_estimators=100)
scores_2 = cross_val_score(rf, X_train,y_train, cv=10, scoring = "accuracy")


rf = DecisionTreeClassifier()
scores_1 = cross_val_score(rf,  X_train,y_train, cv=10, scoring = "accuracy")


rf = SVC(kernel='rbf',random_state=42)
scores_3 = cross_val_score(rf,  X_train,y_train, cv=10, scoring = "accuracy")


rf = SVC(kernel='poly',degree=3,random_state=42)
scores_4 = cross_val_score(rf,  X_train,y_train, cv=10, scoring = "accuracy")


rf = KNeighborsClassifier(n_neighbors = 3)
scores_5 = cross_val_score(rf,  X_train,y_train, cv=10, scoring = "accuracy")


rf = LogisticRegression()
scores_6 = cross_val_score(rf,  X_train,y_train, cv=10, scoring = "accuracy")

a = []
b = []

for i in [scores_1,scores_2,scores_3,scores_4,scores_5,scores_6]:
    a.append(i.mean())
    b.append(i.std())

a = np.array(a)
b = np.array(b)

df = pd.DataFrame([a,b,results[:6].values]).T
df.columns = ['mean_of_cross_val','std_of_cross_val','model_without_cro_ssval']
df.index = results[:6].index

print(df)


# In[ ]:


#shortlisted three models for further imporvement
#svc with rbf, knn, logistic before that let's see the good no. of neighbors of knn


# In[ ]:


"""

just trying to see if i get good results with different value of neighbors

"""


# In[28]:


param_grid = { "n_neighbors" : [1,2,3,4,5,6,7,8],
            "p" : [1,2],
            "leaf_size" : [10,20,30,40,50]
            }
from sklearn.model_selection import GridSearchCV, cross_val_score
rf = KNeighborsClassifier(n_jobs=-1)
clf = GridSearchCV(estimator=rf, param_grid=param_grid,cv = 5,scoring='accuracy',n_jobs=-1)
clf.fit(X_train,y_train)
cvres = clf.cv_results_
cvres['mean_test_score'].max()


# In[29]:


clf.best_params_


# In[30]:


knn = clf.best_estimator_
knn.fit(X_train, y_train) 
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn


# In[ ]:


#now evaluating the first three selected model and testing them on different basis


# In[33]:


def evaluate(a):
    print("confusion matrix\n",confusion_matrix(y_train,a.predict(X_train)))
    print()
    print("precision: ",precision_score(y_train,a.predict(X_train)))
    print("recall: ",recall_score(y_train,a.predict(X_train)))
    print("f1_score: ",f1_score(y_train,a.predict(X_train)))
    print("roc_score: ",roc_auc_score(y_train,a.predict(X_train)))
print("For knn:\n")
evaluate(knn)
print()
print("For svcrbf:\n")
evaluate(svc_rbf)
print()
print("For logistic:\n")
evaluate(logreg)


# In[ ]:


#svm with rbf wins the race


# In[ ]:


#seeing if there is room for improvement


# In[42]:


param_grid = {'C':[0.3,0.5,0.8,1],
            "gamma" : ['auto',0.4,0.8,0.5,0.2],
            "tol" : [1e-3,10e-3],
            "verbose" : [True,False],
              "max_iter":[50,100,150,200,300,-1]
            }
from sklearn.model_selection import GridSearchCV, cross_val_score
rf =  SVC(kernel='rbf',random_state=42)
clf = GridSearchCV(estimator=rf, param_grid=param_grid,cv = 5,scoring='accuracy',n_jobs=-1)
clf.fit(X_train,y_train)


# In[43]:


clf.best_params_


# In[44]:


clf.best_estimator_


# In[45]:


cvres = clf.cv_results_
cvres['mean_test_score'].max()


# In[46]:


clf = clf.best_estimator_
clf.fit(X_train,y_train)


# In[47]:


evaluate(clf)


# In[ ]:


#nothing is changed accuracy increased but that has nothing to do with other evaluation things so back to simple model


# In[48]:


#final model 

svc_rbf = SVC(kernel='rbf',random_state=42,probability=True)
svc_rbf.fit(X_train,y_train)
#round(svc_rbf.score(X_train,y_train)*100,2)


# In[58]:


#y_scores = cross_val_predict(random_forest, train_x,train_y, cv=3,n_jobs=-1)
from sklearn.metrics import precision_recall_curve
y_scores = svc_rbf.predict_proba(X_train)[:,1]
#precision, recall, threshold = precision_recall_curve(train_y, predictions)
#previous i used the above thingy and that gives solid straight line doesn't expected
precision, recall, threshold = precision_recall_curve(y_train, y_scores)


# In[59]:


def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-",label="precision")
    plt.plot(threshold, recall[:-1], "b", label="recall")
    plt.xlabel("threshold")
    plt.legend()
    plt.ylim([0, 1])
plt.figure(figsize=(10, 5))
plot_precision_and_recall(precision, recall, threshold)
    #plt.show()


# In[60]:


def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--")
    plt.ylabel("recall")
    plt.xlabel("precision")
    plt.figure(figsize=(14, 7))
    
plot_precision_vs_recall(precision, recall)
plt.show()


# In[61]:


evaluate(svc_rbf)


# In[ ]:


#i was thinking to change the probability to some other value to improve my results but looks like at this moment it is enough good


# In[ ]:


#still trying with different prob to get good results 


# In[98]:


a=0.7
confusion_matrix(y_train,svc_rbf.predict_proba(X_train)[:,1]>a)


# In[99]:


precision_score(y_train,(svc_rbf.predict_proba(X_train)[:,1]>a))


# In[100]:


recall_score(y_train,svc_rbf.predict_proba(X_train)[:,1]>a)


# In[101]:


f1_score(y_train,svc_rbf.predict_proba(X_train)[:,1]>a)


# In[102]:


roc_auc_score(y_train,svc_rbf.predict_proba(X_train)[:,1]>a)


# In[ ]:


#indeed the results improved now lets finalize the model and check its working on the test set.


# In[103]:


from sklearn.externals import joblib 

# Save the model as a pickle in a file 
joblib.dump(svc_rbf, 'filename1.pkl') 

# Load the model from the file 
clf_from_joblib = joblib.load('filename1.pkl') 

# Use the loaded model to make predictions 
"""clf_from_joblib.predict(test_x) 
"""
clf_from_joblib = joblib.load('filename1.pkl')
#pred = clf_from_joblib.predict(X_test)


# In[117]:


pred = (clf_from_joblib.predict_proba(X_test)>a)[:,1]


# In[118]:


confusion_matrix(y_test,pred)


# In[119]:


precision_score(y_test,pred)


# In[120]:


recall_score(y_test,pred)


# In[121]:


f1_score(y_test,pred)


# In[122]:


roc_auc_score(y_test,pred)


# In[ ]:


df_new = pd.merge(dataset1,dataset,on='Country')[['Country','2015','Value']]


# In[ ]:


dataset = dataset.loc[dataset['INEQUALITY']=='TOT']


# In[ ]:


dataset = pd.pivot_table(dataset,index='Country',columns='Indicator',values="Value")


# In[ ]:


dataset1 = dataset1.set_index('Country')


# In[ ]:


df_new = pd.merge(dataset1,dataset,left_index=True,right_index=True)


# In[ ]:


df_new = df_new.loc[:,['2015','Life satisfaction']].sort_values('2015')


# In[ ]:


df_new = df_new.drop(df_new.iloc[[0, 1, 6, 8, 33, 34, 35]].index)


# In[ ]:


df_new


# In[ ]:


df_new.plot(kind='scatter',y='Life satisfaction',x='2015')


# In[ ]:


X = df_new.iloc[:,0:1].values
y = df_new.iloc[:,1].values


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)


# In[ ]:


model.predict(22587)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X,y)


# In[ ]:


model.predict(22587)


# In[ ]:


df_new.plot(kind='scatter',y='Life satisfaction',x='2015')
plt.plot(X,model.predict(X),c='r')


# # new data

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
housing = pd.read_csv('C:\\Users\\Mridul\\New folder\\handson-ml-master\\datasets\\housing\\housing.csv')


# In[3]:


# Divide by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[4]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[5]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# In[6]:


housing = strat_train_set.copy()


# In[7]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[8]:


housing.describe()


# In[9]:


housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()


# In[10]:


sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows


# In[11]:


median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
sample_incomplete_rows


# In[12]:


from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")


# In[90]:


housing_num = housing.drop('ocean_proximity', axis=1)


# In[91]:


imputer.fit(housing_num)


# In[92]:


imputer.statistics_


# In[16]:


housing_num.median().values


# In[17]:


X = imputer.transform(housing_num)


# In[18]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index = list(housing.index.values))


# In[19]:


housing_tr.loc[sample_incomplete_rows.index.values]


# In[20]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns)
housing_tr.head()


# In[28]:


housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)


# In[75]:


housing_cat.shape


# In[29]:


ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[76]:


housing_cat_encoded.shape


# In[30]:


cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[31]:


housing_cat_1hot.toarray()


# In[32]:


cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[93]:


from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[99]:


housing_extra_attribs[:5].shape


# In[84]:


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
housing_extra_attribs.head()


# In[85]:


housing_extra_attribs.shape


# In[118]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[119]:


housing_num_tr


# In[120]:


housing_num_tr[:5]


# In[89]:


housing_num_tr.shape


# In[38]:


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# In[39]:


housing_prepared


# In[83]:


housing_prepared[:5]


# In[40]:


housing_prepared.shape


# In[41]:


housing_prepared.shape


# In[42]:


housing_labels.shape


# In[145]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[146]:


# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))


# In[148]:


some_data_prepared


# In[45]:



print("Labels:", list(some_labels))


# In[46]:


some_data_prepared


# In[47]:


some_data_prepared.shape


# In[48]:


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[49]:


from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae


# In[50]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# In[51]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[52]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[53]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# In[54]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[55]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# In[56]:


housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[57]:


from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[58]:


scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()


# In[59]:


from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse


# In[60]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# In[61]:


grid_search.best_params_


# In[62]:


grid_search.best_estimator_


# In[63]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[64]:


pd.DataFrame(grid_search.cv_results_)


# In[65]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)


# In[66]:


cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[67]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[121]:


feature_importances.shape


# In[68]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# In[69]:


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[70]:


final_rmse


# In[71]:


from scipy import stats


# In[72]:


confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)

np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))


# In[73]:


tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)


# In[74]:


zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)


# In[134]:


a = pd.DataFrame(np.arange(1,5).reshape(2,2))


# In[144]:


a


# In[137]:


a.iloc[1,1]=5


# In[140]:


b = a.iloc[1,:]


# In[142]:


b[0] = 999


# In[143]:


b


# In[ ]:


dataset['cool'] = np.ceil(dataset['median_income']/1.5)
dataset['cool'].where(dataset['cool']<5,5.0,inplace=True)


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(dataset,dataset['cool']):
    strat_train = dataset.loc[train_index] 
    strat_test  = dataset.loc[train_index]
for i in (strat_train,strat_test):
    i.drop('cool',axis=1,inplace=True)


# In[ ]:


dataset = strat_train.copy()


# In[ ]:


#dataset.plot(kind='scatter',x='longitude',y='latitude',alpha=0.5,s=dataset['population']/100,label='population',
#            c='median_house_value',cmap=plt.get_cmap('jet'),colorbar=True,figsize=(15,12),sharex=False)
#plt.legend()


# In[ ]:


#pd.plotting.scatter_matrix(dataset[['longitude', 'latitude','median_income','median_house_value']],figsize=(15,12));


# In[ ]:


housing = strat_train.drop('median_house_value',axis=1)
housing_labels = strat_train['median_house_value'].copy()


# In[ ]:


housing.head()


# In[ ]:


housing[housing.isnull().any(axis=1)].head()


# In[ ]:


from sklearn.preprocessing import Imputer,OneHotEncoder,LabelEncoder,StandardScaler
impute = Imputer(strategy='median',axis=0)
housing_u = housing.drop('ocean_proximity',axis=1)
X = impute.fit_transform(housing_u)


# In[ ]:


housing_t = pd.DataFrame(X,columns=housing_u.columns,index=housing_u.index)


# In[ ]:


labele = LabelEncoder()
#chater = housing['ocean_proximity']
encoded = labele.fit_transform(housing['ocean_proximity'])


# In[ ]:


onehot = OneHotEncoder()
onehoted = onehot.fit_transform(encoded.reshape(-1,1)).toarray()


# In[ ]:


onehoted


# In[ ]:


housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['population_per_household'] = housing['population']/housing['households']


# In[ ]:


scale_p = StandardScaler()

