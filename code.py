#!/usr/bin/env python
# coding: utf-8

# In[1]:
#Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error



# In[2]:

#Importing datasets

audi = pd.read_csv('data/audi.csv')
bmw = pd.read_csv('data/bmw.csv')
ford = pd.read_csv('data/ford.csv')
hyundai = pd.read_csv('data/hyundai.csv')
merc = pd.read_csv('data/merc.csv')
skoda = pd.read_csv('data/skoda.csv')
toyota = pd.read_csv('data/toyota.csv')
vauxhall = pd.read_csv('data/vauxhall.csv')
vw = pd.read_csv('data/vw.csv')


# In[3]:

# For each datasets we add the make of the car

audi['make']='Audi'
bmw['make']='BMW'
ford['make']='Ford'
hyundai['make']='Hyundai'
merc['make']='Mercedes'
skoda['make']='Skoda'
toyota['make']='Toyota'
vauxhall['make']='Vauxhall'
vw['make']='VW'


# In[4]:

#We merge datasets into one

cars_data=pd.concat([audi, bmw, ford, hyundai, merc, skoda, toyota, vauxhall, vw])


# In[5]:


cars_data.info()


# In[6]:

#We remove columns which we will not be using

cars_data = cars_data.drop(['tax','tax(£)', 'mpg', 'model'], axis=1)


# In[7]:

#We check for missing values  

missing_val = cars_data.isnull().sum()
print(missing_val)


# In[8]:

#We use descriptive statisticts

cars_data_stats=cars_data.describe(include='all')
print(cars_data_stats)


# In[9]:

#We plot price and mileage correlation scatterplot

plt.figure(figsize=(20,10)) 
sns.scatterplot(cars_data["mileage"], cars_data["price"], hue = cars_data["year"])


# In[10]:

#Price distplot for outliers

sns.distplot(cars_data['price'])


# In[11]:

#Mileage distplot for outliers

sns.distplot(cars_data['mileage'])


# In[12]:

#Splitting into numerical and categorical values

cat_data=cars_data.select_dtypes(exclude=["number"])
num_data=cars_data.select_dtypes(include=["number"])


# In[13]:

#Removing outlirs

idx = np.all(stats.zscore(num_data) < 3, axis=1)


# In[14]:


cars_df = pd.concat([num_data.loc[idx], cat_data.loc[idx]], axis=1)


# In[15]:


cars_df_stats=cars_df.describe(include='all')
print(cars_df_stats)


# In[16]:

#We can see that outliers have been removed

plt.figure(figsize=(20,10)) 
sns.scatterplot(cars_df["mileage"], cars_df["price"], hue = cars_df["year"])


# In[17]:


sns.distplot(cars_df['price'])


# In[18]:


sns.distplot(cars_df['mileage'])


# In[19]:


cars_df['year'].value_counts()


# In[20]:

#We select data from year 2000 as before there is not many variables

cars_df=cars_df[cars_df['year']>2000]
sns.countplot(y=cars_df['year'])


# In[21]:


plt.figure(figsize=(20,10)) 
sns.scatterplot(cars_df["mileage"], cars_df["price"], hue = cars_df["year"])


# In[22]:

#Select only valid engine sizes, as probably 0 were the inputs where user did not put any value

cars_df['engineSize'].value_counts()
cars_df=cars_df[cars_df['engineSize']>0.5]
sns.countplot(y=cars_df['engineSize'])


# In[23]:

#Plotting heatmap

plt.figure(figsize = (5,5))
sns.heatmap(cars_df.corr(), annot = True)


# In[24]:
#We can see that the there is no linearity, but exponential

f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize=(15,3))
ax1.scatter(cars_df['year'],cars_df['price'])
ax1.set_title('Price and Year')
ax2.scatter(cars_df['mileage'],cars_df['price'])
ax2.set_title('mileage and price')
ax3.scatter(cars_df['engineSize'],cars_df['price'])
ax3.set_title('engineSize and price')


# In[25]:

#Convert price to log_price so we can have linearity

log_price= np.log(cars_df['price'])
cars_df['log_price'] = log_price
cars_df = cars_df.drop(['price'],axis=1)


# In[26]:


f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize=(15,3))
ax1.scatter(cars_df['year'],cars_df['log_price'])
ax1.set_title('Price and Year')
ax2.scatter(cars_df['mileage'],cars_df['log_price'])
ax2.set_title('mileage and price')
ax3.scatter(cars_df['engineSize'],cars_df['log_price'])
ax3.set_title('engineSize and price')


# In[27]:

#We get dummies

dummies_car = pd.get_dummies(cars_df,drop_first=True)
dummies_car.head()


# In[28]:
dummies_car.head()

# In[29]:

#Declaring inputs and targets
X=dummies_car.drop(['log_price'], axis=1)
y=dummies_car['log_price']


# In[30]:

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=320)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[31]:


scaler = StandardScaler()


# In[32]:


scaler.fit(X_train)


# In[33]:

#Normalizing
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[34]:

#Linear Regression model
reg=LinearRegression()
reg.fit(X_train, y_train)


# In[35]:


y_pred = reg.predict(X_train)


# In[36]:

#We can see that we have linearity
plt.scatter(y_train, y_pred)
plt.xlabel('Targets (y_train)', size=10)
plt.ylabel('Predictions (y_pred)', size=10)
plt.xlim(6,11)
plt.ylim(6,11)


# In[37]:


sns.distplot(y_train - y_pred)


# In[81]:
    
    
y_test_pred=reg.predict(X_test)

# In[39]:

#We get a scallar
reg.intercept_


# In[40]:

#And also an array
reg.coef_


# In[41]:

#Summary
reg_summary = pd.DataFrame(X.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


# In[42]:

#R2 score
print(r2_score(y_test, y_test_pred)*100)


# In[43]:
#Actuals vs Predictions

plt.figure(figsize=(15,10))
c = [i for i in range(1,len(X_test)+1,1)]
plt.plot(c,y_test,linestyle='-',color='b')
plt.plot(c,y_test_pred,linestyle='-',color='r')
plt.title('Actual (blue) Vs Prediction (red)')
plt.xlabel('Index')
plt.ylabel('Price')
plt.show()


# In[44]:


df_pf=pd.DataFrame(np.exp(y_test_pred), columns=['Prediction'])
df_pf.head()


# In[45]:


y_test = y_test.reset_index(drop=True)
y_test.head()


# In[46]:


df_pf['Target']=np.exp(y_test)
df_pf.head()


# In[47]:


df_pf['Remaining'] = df_pf['Target'] -df_pf['Prediction']


# In[48]:


df_pf['Difference%'] = df_pf['Remaining']/df_pf['Target']*100
df_pf


# In[49]:


df_pf.describe()


# In[50]:

#Ridge Regression
model_ridge = Ridge(alpha=100)


# In[51]:

#Cross Validation
scores_r2 = cross_val_score(model_ridge, X_train, y_train, scoring='r2', cv=10)


# In[52]:


scores_r2


# In[53]:


abs(scores_r2.mean())


# In[54]:


scores_mse = cross_val_score(model_ridge, X_train, y_train, scoring='neg_mean_squared_error', cv=10)


# In[55]:


scores_mse


# In[56]:


abs(scores_mse.mean())


# In[57]:


model_ridge.fit(X_train, y_train)


# In[58]:


y_test_pred_rr = model_ridge.predict(X_test)


# In[59]:


mean_squared_error(y_test, y_test_pred_rr)


# In[60]:


r2_score(y_test, y_test_pred_rr)*100


# In[61]:

#Elastic Net Regression
model_enet = ElasticNet()


# In[62]:

#GridSearchCV parameters
param_grid = {'alpha':[0.1,1,5,10,50,100], 'l1_ratio':[.1, .5, .7, .95, .99, 1]}


# In[63]:

#GridSearchCV
grid_model_r2 = GridSearchCV(estimator=model_enet, param_grid=param_grid, scoring='r2',cv=10, verbose=4)


# In[64]:


grid_model_r2.fit(X_train, y_train)


# In[65]:


grid_model_r2.best_estimator_


# In[66]:


pd.DataFrame(grid_model_r2.cv_results_)


# In[67]:


y_test_pred_enet_r2 = grid_model_r2.predict(X_test)


# In[68]:


r2_score(y_test, y_test_pred_enet_r2)*100


# In[69]:


grid_model_mse = GridSearchCV(estimator=model_enet, param_grid=param_grid, scoring='neg_mean_squared_error',cv=10, verbose=4)


# In[70]:


grid_model_mse.fit(X_train, y_train)


# In[71]:


grid_model_mse.best_estimator_


# In[72]:


y_test_pred_enet_mse = grid_model_mse.predict(X_test)


# In[73]:


mean_squared_error(y_test, y_test_pred_enet_mse)


# In[78]:

#Lasso regression
model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X_train, y_train) 
pred_train_lasso= model_lasso.predict(X_train)
print((mean_squared_error(y_train,pred_train_lasso)))
print(r2_score(y_train, pred_train_lasso)*100)

pred_test_lasso= model_lasso.predict(X_test)
print((mean_squared_error(y_test,pred_test_lasso))) 
print(r2_score(y_test, pred_test_lasso)*100)


# In[82]:

#Final Comparison
print('Log Linear r2: {}'.format(r2_score(y_test, y_test_pred)*100))
print('Log Linear MSE: {}'.format(mean_squared_error(y_test, y_test_pred)))

print('Ridge Regression r2:{}'.format(r2_score(y_test, y_test_pred_rr)*100))
print('Ridge Regression MSE: {}'.format(mean_squared_error(y_test, y_test_pred_rr)))

print('Lasso Regression r2: {}'.format(r2_score(y_test, pred_test_lasso)*100))
print('Lasso Regression MSE: {}'.format(mean_squared_error(y_test, pred_test_lasso)))

print('ElasticNet r2: {}'.format(r2_score(y_test, y_test_pred_enet_r2)*100))
print('ElasticNet MSE: {}'.format(mean_squared_error(y_test, y_test_pred_enet_mse)))


# In[ ]:




