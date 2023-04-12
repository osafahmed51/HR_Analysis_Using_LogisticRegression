#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[20]:


df=pd.read_csv('hr_analytics.csv')
df


# In[22]:


left=df[df.left==1]
left.shape


# In[23]:


retention=df[df.left==0]
retention.shape


# In[24]:


df.groupby('left').mean()


# In[25]:


pd.crosstab(df.salary,df.left).plot(kind='bar')


# In[29]:


subdf=df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
subdf


# In[31]:


salary_dumies=pd.get_dummies(subdf.salary,prefix='salary')
salary_dumies


# In[40]:


df_with_dumies=pd.concat([subdf,salary_dumies],axis=1)
df_with_dumies


# In[42]:


df_with_dumies.drop('salary',axis=1,inplace=True)


# In[43]:


df_with_dumies.head()


# In[45]:


X=df_with_dumies
X


# In[47]:


Y=df.left
Y


# In[48]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.3)


# In[49]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[50]:


model.fit(X_train,Y_train)


# In[51]:


model.predict(X_test)


# In[52]:


model.score(X_test,Y_test)


# In[ ]:




