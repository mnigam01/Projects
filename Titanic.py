
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron


# In[4]:


train = pd.read_csv('E:\\train.csv')
test = pd.read_csv('E:\\test.csv')


# In[5]:


train.info()


# In[6]:


train.describe()


# In[7]:


#percentage of people survived the titanic in train
train['Survived'].sum()/len(train['Survived'])*100


# In[8]:


train.head()


# In[9]:


total = train.isnull().sum().sort_values(ascending=False)
percent = total/len(train)*100
percent = round(percent,1)


# In[10]:


df = pd.DataFrame([total,percent]).T
df.columns = ['Total','Percent']
df


# In[11]:


print(train.columns)
print(len(train.columns))


# In[12]:


#(train.groupby(['Sex','Survived']))['Age'].hist(alpha=0.2)


# In[13]:


#train.groupby(['Sex','Survived'])['Age'].describe()


# In[14]:


male_dead = train.set_index(['Sex','Survived']).loc['male',0]['Age'].values
male_survived = train.set_index(['Sex','Survived']).loc['male',1]['Age'].values


# In[15]:


female_dead = train.set_index(['Sex','Survived']).loc['female',0]['Age'].values
female_survived = train.set_index(['Sex','Survived']).loc['female',1]['Age'].values


# In[16]:


fig,ax = plt.subplots(2)
df = pd.DataFrame([male_dead,male_survived]).T
ax[0].hist(df[0],alpha=0.3,label = 'not survived',bins=25)
ax[0].hist(df[1],alpha=0.2,label = 'survived',bins=25)
ax[0].legend()
ax[0].set_title('male')

df = pd.DataFrame([female_dead,female_survived]).T
ax[1].hist(df[0],alpha=0.3,label = 'not survived',bins=25)
ax[1].hist(df[1],alpha=0.2,label = 'survived',bins=25)
ax[1].legend()
ax[1].set_title('female')


# In[17]:


FacetGrid = sns.FacetGrid(train,'Embarked',size=4.5,aspect=2)
FacetGrid.map(sns.pointplot,'Pclass', 'Survived', 'Sex')
FacetGrid.add_legend()


# In[18]:


fig,ax = plt.subplots(3)
sns.barplot(x='Pclass',y='Survived',data = train, ax = ax[0])
sns.barplot(x='Sex',y='Survived',data = train, ax = ax[1])
sns.barplot(x='Embarked',y='Survived',data = train, ax = ax[2])
fig.set_figwidth(10)
fig.set_figheight(10)


# In[19]:


data = [train,test]
for i in data:
    i['relatives'] = i['SibSp'] + i['Parch']
    i.loc[i['relatives']>0,'not_alone']=0
    i.loc[i['relatives']==0,'not_alone']=1


# In[20]:


train['not_alone'].value_counts()


# In[21]:


sns.pointplot('relatives','Survived',data=train,aspect=2.5)


# In[22]:


train.drop('PassengerId',axis=1,inplace=True)
test.drop('PassengerId',axis=1,inplace=True)


# In[23]:


#train.loc[train['relatives']==3,'Survived'].sum()


# In[24]:


#train.loc[train['Survived']==1,'relatives'].value_counts()


# In[25]:


#train.loc[train['Survived']==0,'relatives'].value_counts()


# In[26]:


train_df = train.copy()
test_df = test.copy()


# In[27]:


train_df['Cabin'] = train_df['Cabin'].str[:1]
train_df['Cabin'].fillna('U',inplace=True)


# In[28]:


train_df['Cabin'].value_counts()


# In[29]:


data = [train_df, test_df]
for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and    is_null
    np.random.seed(42)
    rand_age = np.random.randint(mean - std, mean + std,
    size = is_null)
    # fill NaN values in Age column with random values    generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()


# In[30]:


train_df['Embarked'].describe()


# In[31]:


#since two places then fill by using S
for data in [train_df,test_df]:
    data['Embarked'].fillna('S',inplace=True)


# In[32]:


train_df.info()


# In[33]:


test_df['Cabin'] = test_df['Cabin'].str[:1]
test_df['Cabin'].fillna('U',inplace=True)


# In[34]:


test_df.info()


# In[35]:


for dataset in [train_df,test_df]:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[36]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2}
for dataset in [train_df,test_df]:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset.Title.fillna(3,inplace=True)


# In[37]:


for dataset in [train_df,test_df]:
    dataset.drop(['Name'],axis=1,inplace=True)


# In[38]:


train_df.Ticket.nunique()


# In[39]:


#train_df.Ticket
#no correlation can be seen there
#so dropping this column as well
for dataset in [train_df,test_df]:
    dataset.drop(['Ticket'],axis=1,inplace=True)


# In[40]:


from sklearn.preprocessing import LabelEncoder
label_1 = LabelEncoder()
train_df['Sex'] = label_1.fit_transform(train_df['Sex'])
test_df['Sex'] = label_1.transform(test_df['Sex'])

label_2 = LabelEncoder()
train_df['Cabin'] = label_2.fit_transform(train_df['Cabin'])
test_df['Cabin'] = label_2.transform(test_df['Cabin'])

label_3 = LabelEncoder()
train_df['Embarked'] = label_3.fit_transform(train_df['Embarked'])
test_df['Embarked'] = label_3.transform(test_df['Embarked'])


# In[41]:


train_df.head()


# In[42]:


train = train_df.copy()
test = test_df.copy()


# In[43]:


train_y = train_df['Survived'].copy()
train_x = train_df.drop('Survived',axis=1).copy()


# In[44]:


#changing the title thingy into dummy variables
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
encodething = onehot.fit_transform(train_x['Title'].values.reshape(-1,1)).toarray()


# In[45]:


train_x.drop('Title',axis=1,inplace=True)
train_x = pd.concat([train_x,pd.DataFrame(encodething)],axis=1)


# In[46]:


test_x = test_df.copy()
encodething = onehot.transform(test_x['Title'].values.reshape(-1,1)).toarray()
test_x.drop('Title',axis=1,inplace=True)
test_x = pd.concat([test_x,pd.DataFrame(encodething)],axis=1)


# In[47]:


test_x[test_x['Fare'].isnull()]


# In[48]:


print(train_x.Fare.median())
print(test_x.Fare.median())
print(train_x.Fare.skew())
print(test_x.Fare.skew())
#one thing is clear after seeing skewness use median


# In[49]:


#train_x.groupby('Pclass')['Fare'].median()


# In[50]:


#train_x.groupby('Cabin')['Fare'].median()


# In[51]:


#train_x.groupby('Embarked')['Fare'].median()


# In[52]:


#train_x.groupby(['Pclass','Embarked','Cabin'])['Fare'].median()


# In[53]:


#after seeing above scenario beter to use 8.05 as missing value
test_x['Fare'].fillna(train_x.groupby('Pclass')['Fare'].transform('median'),inplace=True)


# In[54]:


features = train_x.columns


# In[55]:


from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler()
train_x = scale_x.fit_transform(train_x)
test_x = scale_x.transform(test_x)


# In[56]:


sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(train_x, train_y)
pred_1 = sgd.predict(test_x)
sgd.score(train_x, train_y)
acc_sgd = round(sgd.score(train_x, train_y) * 100, 2)


# In[57]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_x, train_y)
pred_2 = random_forest.predict(test_x)
random_forest.score(train_x, train_y)
acc_random_forest = round(random_forest.score(train_x, train_y) * 100, 2)


# In[58]:


logreg = LogisticRegression()
logreg.fit(train_x, train_y)
pred_3 = logreg.predict(test_x)
acc_log = round(logreg.score(train_x, train_y) * 100, 2)


# In[59]:


# KNN 
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_x, train_y) 
pred_4 = knn.predict(test_x)
acc_knn = round(knn.score(train_x, train_y) * 100, 2)


# In[60]:


gaussian = GaussianNB() 
gaussian.fit(train_x, train_y)
pred_5 = gaussian.predict(test_x) 
acc_gaussian = round(gaussian.score(train_x, train_y) * 100, 2)


# In[61]:


perceptron = Perceptron(max_iter=5)
perceptron.fit(train_x, train_y)
pred_6 = perceptron.predict(test_x)
acc_perceptron = round(perceptron.score(train_x, train_y) * 100, 2)


# In[62]:


linear_svc = LinearSVC()
linear_svc.fit(train_x, train_y)
pred_7 = linear_svc.predict(test_x)
acc_linear_svc = round(linear_svc.score(train_x, train_y) * 100, 2)


# In[63]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_x, train_y) 
pred_8 = decision_tree.predict(test_x) 
acc_decision_tree = round(decision_tree.score(train_x, train_y) * 100, 2)


# In[64]:


results = pd.Series( [acc_linear_svc, acc_knn, acc_log,
acc_random_forest, acc_gaussian,
acc_perceptron,
acc_sgd, acc_decision_tree],index =  ['Support Vector Machines', 'KNN', 'Logistic Regression',
'Random Forest', 'Naive Bayes', 'Perceptron',
'Stochastic Gradient Decent',
'Decision Tree'])


# In[65]:


results.sort_values(ascending=False)


# In[2]:


from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, train_x, train_y, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[67]:


rf = DecisionTreeClassifier()
scores_1 = cross_val_score(rf, train_x, train_y, cv=10, scoring = "accuracy")
print("Scores:", scores_1)
print("Mean:", scores_1.mean())
print("Standard Deviation:", scores_1.std())


# In[68]:


importances = pd.Series(np.round(random_forest.feature_importances_,3),index=features)
importances = importances.sort_values(ascending=False)
importances.plot.bar()


# In[69]:


train_x = pd.DataFrame(train_x,columns=features)
test_x = pd.DataFrame(test_x,columns=features)


# In[70]:


train_df = train_x.copy()
test_df = test_x.copy()


# In[71]:


#removing less useful features from the dataset
train_x = train_x.drop(["not_alone",3,'Parch',2], axis=1)
test_x = test_df.drop(["not_alone",3,'Parch',2], axis=1)


# In[72]:


random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(train_x, train_y)
Y_prediction = random_forest.predict(test_x)
random_forest.score(train_x, train_y)
acc_random_forest = round(random_forest.score(train_x, train_y) * 100, 2)
print(round(acc_random_forest,2,), "%")


# In[73]:


print("oob score:", round(random_forest.oob_score_, 4)*100,"%")


# In[74]:


from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, train_x, train_y, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[82]:


param_grid = { "criterion" : ["gini", "entropy"],
"min_samples_leaf" : [1, 5, 10, 25, 50, 70],
"min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35],
"n_estimators": [100, 400, 700, 1000, 1500]}
from sklearn.model_selection import GridSearchCV, cross_val_score
rf = RandomForestClassifier(max_features='auto',oob_score=True, random_state=0,n_jobs=-1)
clf = GridSearchCV(estimator=rf, param_grid=param_grid,n_jobs=-1)
clf.fit(train_x, train_y)


# In[75]:


from sklearn.externals import joblib 

# Save the model as a pickle in a file 
joblib.dump(clf, 'filename.pkl') 

# Load the model from the file 
clf_from_joblib = joblib.load('filename.pkl') 

# Use the loaded model to make predictions 
"""clf_from_joblib.predict(test_x) 
"""


# In[85]:


clf.best_params_


# In[86]:


bestestimator = clf.best_estimator_


# In[87]:


random_forest = bestestimator
random_forest.fit(train_x,train_y)
Y_prediction = random_forest.predict(test_x)
random_forest.score(train_x,train_y)
print("oob score:", round(random_forest.oob_score_, 4)*100,"%")


# In[88]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, train_x,train_y, cv=3,n_jobs=-1)
confusion_matrix(train_y, predictions)


# In[89]:


from sklearn.metrics import precision_score, recall_score
print("Precision:", precision_score(train_y, predictions))
print("Recall:",recall_score(train_y, predictions))


# In[90]:


from sklearn.metrics import f1_score
f1_score(train_y, predictions)


# In[91]:


#y_scores = cross_val_predict(random_forest, train_x,train_y, cv=3,n_jobs=-1)
from sklearn.metrics import precision_recall_curve
y_scores = random_forest.predict_proba(train_x)
y_scores = y_scores[:,1]
#precision, recall, threshold = precision_recall_curve(train_y, predictions)
#previous i used the above thingy and that gives solid straight line doesn't expected
precision, recall, threshold = precision_recall_curve(train_y, y_scores)


# In[92]:


def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-",label="precision")
    plt.plot(threshold, recall[:-1], "b", label="recall")
    plt.xlabel("threshold")
    plt.legend()
    plt.ylim([0, 1])
plt.figure(figsize=(10, 5))
plot_precision_and_recall(precision, recall, threshold)
    #plt.show()


# In[93]:


def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--")
    plt.ylabel("recall")
    plt.xlabel("precision")
    plt.figure(figsize=(14, 7))
    
plot_precision_vs_recall(precision, recall)
plt.show()


# In[94]:


from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds =roc_curve(train_y, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate,label=None):
    plt.plot(false_positive_rate, true_positive_rate,label=label)
    plt.plot([0, 1], [0, 1], 'r')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
plt.figure(figsize=(10,5))
plot_roc_curve(false_positive_rate, true_positive_rate)
#plt.show()


# In[95]:


from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(train_y, y_scores)
print("ROC-AUC-Score:", r_a_score)


# In[96]:


print("This is the prediction result of: \n",Y_prediction)


# In[1]:


Y_prediction


# In[78]:


clf_from_joblib = joblib.load('filename.pkl')
pred = clf_from_joblib.predict(test_x)


# In[87]:


df = pd.DataFrame([np.arange(892,len(pred)+892),pred]).T


# In[89]:


df.columns = ['PassengerId','Survived']


# In[95]:


df.to_csv('sol.csv',index=False)


# In[93]:


pwd

