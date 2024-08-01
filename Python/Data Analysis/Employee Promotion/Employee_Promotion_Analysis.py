#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import  confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV,RepeatedStratifiedKFold,ShuffleSplit,cross_val_score
from sklearn.metrics import accuracy_score, plot_confusion_matrix,f1_score,roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier


# In[2]:


# Load dataset
df = pd.read_csv('train.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


# check missing values
df.isnull().any()


# In[7]:


print('The number of missing value data in the "education" column :',df['education'].isnull().sum())
print('The percentage of number of missing value data in the "education" column :',(df['education'].isnull().sum())*100/df.shape[0],'%')


# In[8]:


print('The number of missing value data in the "pendidikan" column :',df['previous_year_rating'].isnull().sum())
print('The percentage of number of missing value data in the "pendidikan" column :',(df['previous_year_rating'].isnull().sum())*100/df.shape[0],'%')


# Because the missing value is relatively small (< 10%), we don't need to throw it away. We try to do imputation with the average value or mode of each column that has a missing value.

# In[7]:


# Fill in the missing value in the education column with the value that appears the most, namely Bachelor's

df['education']=df['education'].fillna("Bachelor's")
df['education'].count()


# In[8]:


# Fill in the missing value in the 'previous_year_rating' column with the average value or mode

df['previous_year_rating']=df['previous_year_rating'].fillna(3.0)
df['previous_year_rating'].count()


# In[11]:


df.info()


# In[9]:


# Check duplicates
df.duplicated().any()


# In[10]:


# Check outliers
df_num = df[["no_of_trainings","age","length_of_service","avg_training_score"]]


# In[15]:


fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 12))
index = 0
axs = axs.flatten()
for k,v in df_num.items():
    sns.boxplot(y=k, data=df_num, ax=axs[index], orient="H")
    index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[11]:


display(df.describe().loc[["mean","50%","std"]].loc[:,["no_of_trainings","age",
                                                       "length_of_service","avg_training_score"]])


# => It looks like we have outlier in 'length_of_service' feature. We also have outlier in 'age' feature, but it looks like natural outlier. We will handle outliers in data preprocessing process.

# In[ ]:


# EXPLORATORY DATA ANALYSIS


# In[12]:


# Department feature
sns.set_theme(style="whitegrid")
sns.set(rc = {'figure.figsize':(13,8)})
ax = sns.countplot(x='department',data=df)


# => It can be seen from the graph above that the Sales & marketing, Operations, and Procurement dominate the data in this department feature.

# In[13]:


sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='department',hue='is_promoted',data=df,kind="count",height=8, aspect=1.5)


# From the graph above, it can be seen that employees from the Sales & marketing, operations, and Technology departments were promoted the most.

# In[14]:


# Education feature
sns.set_theme(style='whitegrid')
sns.set(rc = {'figure.figsize':(13,8)})
df['education'].value_counts().plot(kind="pie", autopct="%.0f%%")
plt.show()


# In[22]:


sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='education',hue='is_promoted',data=df,kind="count",height=8, aspect=0.8)


# => employees with bachelor's and master's educational backgrounds are the most promoted.

# In[15]:


# Gender feature
sns.set_theme(style='whitegrid')
sns.set(rc = {'figure.figsize':(13,8)})
df['gender'].value_counts().plot(kind="pie", autopct="%.0f%%")
plt.show()


# => the data is dominated by Male employees

# In[16]:


sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='gender',hue='is_promoted',data=df,kind="count",height=8, aspect=0.8)


# => male employees are also promoted more than female (in number).

# In[17]:


# Recruitment feature
sns.set_theme(style='whitegrid')
sns.set(rc = {'figure.figsize':(13,8)})
df['recruitment_channel'].value_counts().plot(kind="pie", autopct="%.0f%%")
plt.show()


# => most of the employee recruitment paths are "other" or not clearly described. The second position is mostly occupied by "sourcing".

# In[18]:


sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='recruitment_channel',hue='is_promoted',data=df,kind="count",height=8, aspect=0.8)


# => employees with "other" recruitment processes or not clearly described occupy the most positions in the order of promotion. While employees with "referred" recruitment paths are almost non-existent (slightly) are promoted.

# In[19]:


# Number of training
sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='no_of_trainings',hue='is_promoted',data=df,kind="count",height=8, aspect=2)


# => the employees with the least amount of training (i.e. 1) are actually promoted the most.

# In[20]:


# Age
sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='age',hue='is_promoted',data=df,kind="count",height=8, aspect=2)


# => employees have age from 27-35 years are the most promoted employees.

# In[21]:


# previous_year_rating
sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='previous_year_rating',hue='is_promoted',data=df,kind="count",height=8, aspect=2)


# => employees with a rating of 5 are the most promoted employees.

# In[22]:


# length_of_service or working period
sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='length_of_service',hue='is_promoted',data=df,kind="count",height=8, aspect=2)


# => employees with 2-4 years of service are the most promoted employees.

# In[27]:


# Awards awards_won?
sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='awards_won?',hue='is_promoted',data=df,kind="count",height=8, aspect=2)


# => employees who do not have awards are the most promoted because the number of employees who do not have awards is much higher than employees who have awards.

# In[26]:


# avg_training_score
sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='avg_training_score',hue='is_promoted',data=df,kind="count",height=8, aspect=2)


# => approximately employees with average training scores in the range 59-62 and 80-85 were the most promoted (although the difference may be small).

# In[28]:


# promoted
sns.set_theme(style='whitegrid')
sns.set(rc = {'figure.figsize':(13,8)})
df['is_promoted'].value_counts().plot(kind="pie", autopct="%.0f%%")
plt.show()


# => our target feature data between positive and negative is not balanced. We have more negative data.

# In[29]:


# Check correlation of features in data
plt.figure(figsize=(20,15))
# plot heat map
g=sns.heatmap(df.corr(),annot=True,cmap="RdYlGn")


# => there is no significant and significant correlation of each feature. It means that we don't have multicollinearity problem.

# In[ ]:


# DATA PREPROCESSING


# In[30]:


# drop some columns not needed
new_df = df.copy()


# In[31]:


new_df.shape


# In[32]:


df = df.drop(["employee_id","region"],axis=1)


# In[33]:


df.head()


# In[34]:


# remove outliers
df.describe()


# In[35]:


df[df['no_of_trainings']==10]


# In[36]:


df['no_of_trainings'].value_counts()


# Looks like the value 10 in the training_number column still makes sense. It doesn't get thrown away. Then, the working period column will then be checked for possible outliers.

# In[37]:


df[df['length_of_service']==37]


# In[40]:


# IQR Method
Q12 = df['length_of_service'].quantile(0.25)
Q32 = df['length_of_service'].quantile(0.75)
IQR2 = Q32-Q12


# In[44]:


Q32+(1.5*IQR2)


# In[46]:


df[df['length_of_service']>(Q32+(1.5*IQR2))]


# It can be seen that data with tenure of more than 14 years includes outliers based on the IQR formula. However here I am discarding only data with values 20 years and over.

# In[45]:


df['length_of_service'].value_counts()


# In[46]:


df= df[~((df['length_of_service']>21))]


# In[47]:


df.shape


# In[48]:


df.info()


# In[49]:


# Categorical encoding
def one_hot_encoder(data,feature,keep_first=True):

    one_hot_cols = pd.get_dummies(data[feature])
    
    for col in one_hot_cols.columns:
        one_hot_cols.rename({col:f'{feature}_'+col},axis=1,inplace=True)
    
    new_data = pd.concat([data,one_hot_cols],axis=1)
    new_data.drop(feature,axis=1,inplace=True)
    
    if keep_first == False:
        new_data=new_data.iloc[:,1:]
    
    return new_data


# In[50]:


df_onehot=df.copy()
for col in df_onehot.select_dtypes(include='O').columns:
    df_onehot=one_hot_encoder(df_onehot,col)


# In[51]:


df_onehot.head()


# In[55]:


df_onehot.info()


# In[52]:


# Features selection
from sklearn.feature_selection import chi2

X_chi = df_onehot.drop(['no_of_trainings','age','previous_year_rating','length_of_service','awards_won?','avg_training_score','is_promoted'],axis=1)
y_chi = df_onehot['is_promoted']


# In[53]:


X_chi.columns


# In[54]:


chi_score = chi2(X_chi,y_chi)
chi_score


# In[55]:


p_values = pd.Series(chi_score[1],index = X_chi.columns)
p_values.sort_values(ascending = False , inplace = True)


# In[61]:


p_values.plot.bar()


# In[62]:


df_onehot.columns


# In[56]:


df_onehot = df_onehot.drop(['recruitment_channel_sourcing', 'education_Below Secondary', 'recruitment_channel_other',
                           'department_Finance', 'gender_m', 'gender_f', 'department_R&D',
                            'department_Operations'],axis=1)


# In[57]:


df_onehot.shape


# In[58]:


# Dimensional reduction with PCA
df2 = df_onehot.copy()


# In[59]:


x_pc = df2.drop('is_promoted',axis = 1).values
y_pc = df2['is_promoted'].values


# In[60]:


x_pc = StandardScaler().fit_transform(x_pc)
x_pc = pd.DataFrame(x_pc)


# In[61]:


x_pc.head()


# In[62]:


from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(x_pc)
x_pca = pd.DataFrame(x_pca)
x_pca.head()


# In[63]:


explained_variance = pca.explained_variance_ratio_
explained_variance


# In[64]:


x_pca['is_promoted']=y_pc
x_pca.columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','is_promoted']
x_pca.head()


# In[65]:


pca2 = PCA(n_components=8)


# In[66]:


Xp = df2.drop('is_promoted',axis = 1).values
yp = df2['is_promoted'].values


# In[67]:


pca2.fit(Xp)
# apply transform to dataset
Xp = pca2.transform(Xp)


# In[68]:


Xp.shape


# In[69]:


# Split into training and test data
scaler = StandardScaler()
X = df_onehot.drop("is_promoted",axis=1)
y = df_onehot["is_promoted"]


# In[70]:


X_train, X_test, y_train, y_test = train_test_split(Xp, yp, test_size = 0.2, stratify=yp)


# In[71]:


# Scaling
scaler.fit(X_train, y_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


# MACHINE LEARNING MODELS


# In[72]:


# KNN
model_knn = KNeighborsClassifier()
model_knn.fit(X_train,y_train)


# In[73]:


pred_knn = model_knn.predict(X_test)
print(classification_report(y_test,pred_knn))


# In[74]:


def Confusion_Matrix(y_test,ypred):
    cfmat = confusion_matrix(y_test,ypred)
    print('Confusion Matrix: \n',classification_report(y_test,ypred,labels=[0,1]))
    print("\n")
    print('TN - True Negative {}'.format(cfmat[1,1]))
    print('FP - False Positive {}'.format(cfmat[1,0]))
    print('FN - False Negative {}'.format(cfmat[0,1]))
    print('TP - True Positive {}'.format(cfmat[0,0]))
    print('Accuracy Rate: {}'.format(np.divide(np.sum([cfmat[0,0],cfmat[1,1]]),np.sum(cfmat))))
    print('Misclassification Rate: {}'.format(np.divide(np.sum([cfmat[0,1],cfmat[1,0]]),np.sum(cfmat))))
    print('F1-Score: {}'.format(f1_score(y_test, ypred,average='macro')))
    print('ROC-AUC {}'.format(roc_auc_score(y_test,ypred)))
    
Confusion_Matrix(y_test,pred_knn)


# In[75]:


# Naive-Bayes
model_gaussian = GaussianNB()
model_gaussian.fit(X_train,y_train)


# In[76]:


pred_gauss = model_gaussian.predict(X_test)
print(classification_report(y_test,pred_gauss))


# In[77]:


Confusion_Matrix(y_test,pred_gauss)


# In[78]:


# Logistic regression
model_LR = LogisticRegression()
model_LR.fit(X_train,y_train)


# In[79]:


pred_LR = model_LR.predict(X_test)
print(classification_report(y_test,pred_LR))


# In[80]:


Confusion_Matrix(y_test,pred_LR)


# In[81]:


# Random forest
model_forest = RandomForestClassifier()
model_forest.fit(X_train,y_train)


# In[83]:


pred_forest = model_forest.predict(X_test)
print(classification_report(y_test,pred_forest))


# In[84]:


Confusion_Matrix(y_test,pred_forest)


# As noted in the target visualization section we have ("promoted") number is not balanced between positive and negative classes. Therefore, we should not rely solely on accuracy metrics, but we also need to look at precision metrics, recall, and most importantly, F1-Score which combines the two.
# 
# It seems that all the models produce quite high accuracy (almost all above 90%), but we need to select the model based on its F1-Score as well. From several models, it was obtained that the highest F1-Score was generated by the Random Forest model, which was around 70%. Then in this case, Random Forest is our optimal model. However we can improve this model by using smote algorithm to oversample and also we can undersample our data.

# In[ ]:


# MODELS USING UNDERSAMPLED & OVERSAMPLED DATA


# In[85]:


df_onehot.head()


# In[86]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from numpy import where
from collections import Counter


# In[87]:


X = df_onehot.drop("is_promoted",axis=1)
y = df_onehot["is_promoted"]


# In[88]:


counter = Counter(y)
print(counter)


# This is our target class before using smote and undersampling

# In[89]:


# define pipeline
strategy1 = {0:43000}
strategy2 = {1:43000}
over = SMOTE(sampling_strategy=strategy2)
under = RandomUnderSampler(sampling_strategy=strategy1)

steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)


# In[90]:


# transform the dataset
X_new, y_new = pipeline.fit_resample(Xp, yp)
# summarize the new class distribution
counter_new = Counter(y_new)
print(counter_new)


# => This is our target class after using smote and undersampling. Maybe they are not so much equal, but at least we have better proportion now.

# In[91]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(X_new,y_new,test_size = 0.2,stratify=y_new)


# In[92]:


scaler.fit(X_train2, y_train2)
X_train2 = scaler.transform(X_train2)
X_test2 = scaler.transform(X_test2)


# In[93]:


# KNN model after SMOTE and undersampled data
model_knn2 = KNeighborsClassifier()
model_knn2.fit(X_train2,y_train2)


# In[94]:


pred_knn2 = model_knn2.predict(X_test)
Confusion_Matrix(y_test,pred_knn2)


# In[95]:


# Naive Bayes
model_gaussian2 = GaussianNB()
model_gaussian2.fit(X_train2,y_train2)


# In[96]:


pred_gauss2 = model_gaussian2.predict(X_test)
Confusion_Matrix(y_test,pred_gauss2)


# In[97]:


# Logistic regression
model_LR2 = LogisticRegression()
model_LR2.fit(X_train2,y_train2)


# In[98]:


pred_LR2 = model_LR2.predict(X_test)
Confusion_Matrix(y_test,pred_LR2)


# In[99]:


# Random forest
model_forest2 = RandomForestClassifier()
model_forest2.fit(X_train2,y_train2)


# In[100]:


pred_forest2 = model_forest2.predict(X_test)
Confusion_Matrix(y_test,pred_forest2)


# In[101]:


model_forest2.predict_proba(X_test)[:,0]


# The results maybe are not as good as our models before we used SMOTE. But at least at this stage i am quite sure that our models are not too overfitting as now we have almost balanced target class. As we can see our F1-Score values are increasing.
# We have the best score from random forest model which i am pretty sure it can be improved by doing hyperparameter tuning.

# In[102]:


# GridSearchCV model
def best_model(X,y):
    models={
        'K_Nearest_Neighbors':{
            'model': KNeighborsClassifier(),
            'params':{'n_neighbors': [5,10,15,20], 'weights': ['uniform','distance'],
                     'algorithm': ['auto', 'ball_tree', 'kd_tree']}
        },
        'Decision_Tree':{
            'model': DecisionTreeClassifier(),
            'params':{'criterion': ['gini', 'entropy'], 'splitter': ['best','random'],
                     'max_depth': [1,2,10,15],'max_leaf_nodes':[2,5,10,15]}
        },
        'Logistic_Regression':{
            'model': LogisticRegression(),
            'params':{'penalty': ['l1', 'l2', 'none'], 
                      'solver': ['newton-cg', 'lbfgs', 'liblinear'],
                     'max_iter': [20,40,50]}
        },
        'Random_Forest':{
            'model': RandomForestClassifier(),
            'params':{'n_estimators':[50,100,150,200],
                      'criterion': ['gini', 'entropy'], 'max_features': ['auto','sqrt','log2'],
                     'max_depth': [5,7,10],'max_leaf_nodes':[5,10,15]}
        }}
    
    scores=[]
    cv = ShuffleSplit(n_splits=5, test_size=0.2,random_state=0)
    for model_name,config in models.items():
        gs = GridSearchCV(config['model'],config['params'],cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'Model':model_name,
            'Best_Score':gs.best_score_,
            'Best_Params':gs.best_params_
        })
        
    return pd.DataFrame(scores, columns=['Model','Best_Score','Best_Params'])


# In[105]:


# Logistic Regression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ['newton-cg', 
                                                                                                 'lbfgs', 
                                                                                                 'liblinear']}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train2, y_train2)


# In[106]:


pred_gridlogreg = grid_log_reg.best_estimator_.predict(X_test)
Confusion_Matrix(y_test,pred_gridlogreg)


# In[ ]:


# KNN
knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                 'weights': ['uniform','distance']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train2, y_train2)


# In[ ]:


pred_gridknn = grid_knears.best_estimator_.predict(X_test)
Confusion_Matrix(y_test,pred_gridknn)

