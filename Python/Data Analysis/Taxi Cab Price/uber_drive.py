# Inspiration
#Some ideas worth exploring:

#What is the average length of the trip?

#Average number of rides per week or per month?

#Total tax savings based on traveled business miles?

#Percentage of business miles vs personal vs. Meals

#How much money can be saved by a typical customer using Uber, Careem, or Lyft versus regular cab service?

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

uber_data = pd.read_csv('My_Uber_Drives - 2016.csv')
print(uber_data.head())

# Handle unwanted data:
print(uber_data.isnull().sum())
uber_data = uber_data.dropna()

# make str as datetime to identify individual components easily
uber_data['START_DATE*'] = pd.to_datetime(uber_data['START_DATE*'], format = "%m/%d/%Y %H:%M")
uber_data['END_DATE*'] = pd.to_datetime(uber_data['END_DATE*'], format = "%m/%d/%Y %H:%M")
print(uber_data.head())

# for x in uber_data['START_DATE*']:
# 	print(x.dayofweek)
uber_data['HOUR'] = [x.hour for x in uber_data['START_DATE*']]
uber_data['DAY'] = [x.day for x in uber_data['START_DATE*']]
uber_data['MONTH'] = [x.month for x in uber_data['START_DATE*']]
uber_data['WEEKDAY'] = [calendar.day_name[x.dayofweek] for x in uber_data['START_DATE*']]
uber_data['DAY_OF_WEEK'] = [x.dayofweek for x in uber_data['START_DATE*']]

print(uber_data.head())

# Count the freq of every unique value:
# sns.countplot(x='CATEGORY*', data = uber_data)
# Users most used Uber for work-related meetings and meals most of the time
# sns.countplot(x='PURPOSE*', data = uber_data)
# Distances traveled by the user are relatively short
# uber_data['MILES*'].plot.hist()
# plt.show()
# See at what time of the day the user rides an Uber the most?
# hours = uber_data['HOUR'].value_counts()
# hours.plot(kind = 'bar', color = 'red', figsize = (10,5))
# plt.xlabel('Hours')
# plt.ylabel('Frequency')
# plt.title('Number of trip VS hours')

# Look at the user's travel patterns on different days of the week.
# days = uber_data['WEEKDAY'].value_counts()
# days.plot(kind = 'barh', color = 'orange', figsize = (10,5))
# plt.xlabel('Days')
# plt.ylabel('Frequency')
# plt.title('Number of trip VS days')
# plt.show()
# User travels almost regularly each day of the week, he travels more on Fridays
# we can also look at the month-wise distribution of Uber trips
# months = uber_data['MONTH'].value_counts()
# months.plot(kind = 'barh', color = 'yellow', figsize = (10,5))
# plt.xlabel('Months')
# plt.ylabel('Frequency')
# plt.title('Number of trip VS months')
# plt.show()
# we must point out how there were significantly more trips in December 2016 for this user . 
# while the rest of the months fall within a specific range?
# see on which days of December the user traveled in an Uber?
# months = uber_data['DAY'][uber_data['MONTH']==12].value_counts()
# months.plot(kind = 'bar', color = 'darkgrey', figsize = (10,5))
# plt.xlabel('Days of December')
# plt.ylabel('Frequency')
# plt.title("Number of trip VS December's day")
# plt.show()
# As expected, the user traveled a lot during the Christmas break
# months = uber_data['START*'].value_counts().nlargest(10)
# months.plot(kind = 'barh', color = 'brown', figsize = (10,5))
# plt.xlabel('Frequency')
# plt.ylabel('Pick up point')
# plt.title("Pick up point VS Freq")
# plt.show()

# ==> The skewed number of trips start from Cary could mean that the user either resides or works in this region. 
# Similarly, let’s also look at the destination of these trips.

# months = uber_data['STOP*'].value_counts().nlargest(10)
# months.plot(kind = 'barh', color = 'grey', figsize = (10,5))
# plt.xlabel('Frequency')
# plt.ylabel('End point')
# plt.title("End VS Freq")
# plt.show()

# An interesting observation is how most of these places are the same as the pick-up points. 
# This confirms the intuition that the user usually commutes around Cary or Morrisville.
df.groupby(by=["destination","source"]).agg({'latitude':'mean','longitude':'mean'})

df_group = df.groupby(by=["source","destination"]).price.agg(["mean"]).reset_index()
df_group[(df_group['source']=='Financial District')& (df_group['destination']=='Fenway')]

new_df = df.drop(['id','timestamp','datetime','long_summary','apparentTemperatureHighTime','apparentTemperatureLowTime',
                  'apparentTemperatureLowTime','windGustTime','sunriseTime','sunsetTime','uvIndexTime','temperatureMinTime',
                 'temperatureMaxTime','apparentTemperatureMinTime','temperatureLowTime','apparentTemperatureMaxTime'],axis=1)

# Our goal is to make linear regression model. First we check correlation between our features and target feature (price)
# First, i want to check the correlation of our temperature related features with our target feature (Price
temp_cols= ['temperature','apparentTemperature','temperatureHigh','temperatureLow','apparentTemperatureHigh',
                'apparentTemperatureLow','temperatureMin','temperatureHighTime','temperatureMax','apparentTemperatureMin','apparentTemperatureMax','price']


df_temp = new_df[temp_cols]
df_temp.head()


plt.figure(figsize=(15,20))
sns.heatmap(df_temp.corr(),annot=True)

# We see that all temperature related features have weak correlation with our target feature which is price¶
# Removing all of them will not make any impact to our regression model

new_df = new_df.drop(['temperature','apparentTemperature','temperatureHigh','temperatureLow','apparentTemperatureHigh',
                'apparentTemperatureLow','temperatureMin','temperatureHighTime','temperatureMax','apparentTemperatureMin','apparentTemperatureMax'],axis=1)
new_df.shape


# Second, i want to check the correlation of our cilmate related features with our target feature (Price)

climate_column = ['precipIntensity', 'precipProbability', 'humidity', 'windSpeed',
       'windGust', 'visibility', 'dewPoint', 'pressure', 'windBearing',
       'cloudCover', 'uvIndex', 'ozone', 'moonPhase',
       'precipIntensityMax','price']
df_clim = new_df[climate_column]
df_clim.head()


plt.figure(figsize=(15,20))
sns.heatmap(df_clim.corr(),annot=True)

# Apparently all climate related features also have weak correlation with our target feature which is price 
# Once again, removing all of them will not make any impact to our regression mode


new_df = new_df.drop(['precipIntensity', 'precipProbability', 'humidity', 'windSpeed',
       'windGust', 'visibility', 'dewPoint', 'pressure', 'windBearing',
       'cloudCover', 'uvIndex', 'ozone', 'moonPhase',
       'precipIntensityMax'],axis=1)
new_df.shape

# Third, i want to check our categorical value in our dataset features

category_col = new_df.select_dtypes(include=['object','category']).columns.tolist()
for column in new_df[category_col]:
    print(f'{column} : {new_df[column].unique()}')
    print()


# We can see that 'timezone' feature has only 1 value and 'product_id' feature contains many unidentified values. So we can remove or drop them.
new_df = new_df.drop(['timezone','product_id'],axis=1)
# Fourth, i want to check the correlation of our categorical features with our target feature (price)


new_cat = ['source',
 'destination',
 'cab_type',
 'name',
 'short_summary',
 'icon','price']

df_cat = new_df[new_cat]
df_cat.head()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df_cat_encode= df_cat.copy()
for col in df_cat_encode.select_dtypes(include='O').columns:
    df_cat_encode[col]=le.fit_transform(df_cat_encode[col])

df_cat_encode.head()

plt.figure(figsize=(15,20))
sns.heatmap(df_cat_encode.corr(),annot=True)

#We can see only name feature that has a relatively strong correlation. Source,destination, and cab_type features have relatively weak correlation, but i will pick cab_type feature because it has stronger correlation than other two features. I will drop or remove the rest of the columns

new_df = new_df.drop(['source','destination','short_summary','icon'],axis=1)
new_df.head()

#Also i will remove hour, day, month, latitude, longitude, because we won't need them for now
new_df = new_df.drop(['hour','day','month','latitude','longitude'],axis=1)
new_df.head()

new_df.columns
# 2. Removing Outliers
#We've already done this before but only to one instance which has maximum price value. We want to check another possible outlier.¶
#We're using IQR method for checking top and bottom outliers

Qp12 = new_df['price'].quantile(0.25)
Qp32 = new_df['price'].quantile(0.75)
IQRp = Qp32-Qp12

new_df[new_df['price']>(Qp32+(1.5*IQRp))]

new_df[new_df['price']<(Qp12-(1.5*IQRp))]
#We can see that we have 5588 data outliers. We can remove or drop them.

print('Size before removing :',new_df.shape)
new_df= new_df[~((new_df['price']>(Qp32+(1.5*IQRp))))]
print('Size after removing :',new_df.shape)

# 4. Regression Model
# 1. Encoding Data (One Hot Encoding)

def one_hot_encoder(data,feature,keep_first=True):

    one_hot_cols = pd.get_dummies(data[feature])
    
    for col in one_hot_cols.columns:
        one_hot_cols.rename({col:f'{feature}_'+col},axis=1,inplace=True)
    
    new_data = pd.concat([data,one_hot_cols],axis=1)
    new_data.drop(feature,axis=1,inplace=True)
    
    if keep_first == False:
        new_data=new_data.iloc[:,1:]
    
    return new_data

new_df_onehot=new_df.copy()
for col in new_df_onehot.select_dtypes(include='O').columns:
    new_df_onehot=one_hot_encoder(new_df_onehot,col)
    
new_df_onehot.head()

#2.Split

from sklearn.model_selection import train_test_split
X = new_df_onehot.drop(columns=['price'],axis=1).values
y = new_df_onehot['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Modeling
from sklearn.linear_model import LinearRegression
reg = LinearRegression() #Base model

# Fit to data training
model = reg.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# Then for the long journey we have done, we got our regression model with accuracy or score 93.37% and RMSE value 2.26. It's not the best score though, we still can improve it with other regression models which could give better results


# Finding Best Models with best configuration with GridSearch CV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV,ShuffleSplit

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

import warnings
warnings.filterwarnings('ignore')

find_best_model_using_gridsearchcv(X,y)


#  Here we got our best model is decision tree regressor with r-squared 0.964, higher than our linear regression before