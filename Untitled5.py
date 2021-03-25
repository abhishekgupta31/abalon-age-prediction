#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import  train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR


# In[2]:


df=pd.read_csv('abalon.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df['age']=df['Rings']+1.5
df_n =df.drop('Rings',axis=1)
df_n.head()


# In[7]:


df['age'].head()


# In[8]:


sns.heatmap(df.isnull())


# In[9]:


sns.pairplot(df)


# In[10]:


n_f = df.select_dtypes(include=[np.number]).columns
c_f = df.select_dtypes(include=[np.object]).columns


# In[11]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sns.heatmap(df[n_f].corr())


# In[12]:


sns.countplot(df['Sex'])


# In[13]:


df.hist(figsize=(10,10))


# In[14]:


df.skew()


# In[15]:


print("Value Count of age Column")
print(df.Rings.value_counts())
print("\nPercentage of age Column")
print(df.Rings.value_counts(normalize = True))


# In[16]:


print('\n Sex count in % ')
print(df.Sex.value_counts(normalize=True))
print('\n Sex count in numbers')
print(df.Sex.value_counts())


# In[17]:


df.groupby('Sex')[['Length','Diameter','Height','Whole weight', 
                   'Shucked weight','Viscera weight', 'Shell weight', 'age']].mean().sort_values(by = 'age',ascending = False)


# In[18]:


from sklearn.preprocessing import LabelEncoder
df['Sex']= LabelEncoder().fit_transform(df['Sex'].tolist())


# In[19]:


from sklearn.preprocessing import OneHotEncoder
trans_sex_ftr = OneHotEncoder().fit_transform(df['Sex'].values.reshape(-1,1)).toarray()
df_sex_enc = pd.DataFrame(trans_sex_ftr, columns = ["Sex_"+str(int(i)) for i in range(trans_sex_ftr.shape[1])])
df = pd.concat([df, df_sex_enc], axis=1)


# In[20]:


df.head()


# In[21]:


x= df.drop(['age','Rings','Sex'],axis=1)
y = df['Rings']


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# In[23]:


print(x.shape)
print(y.shape)


# In[24]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)


# In[25]:


pred = lr.predict(x_test)
pred


# In[26]:


from sklearn.metrics import accuracy_score


# In[27]:


accuracy_score(y_test,pred)


# In[28]:


df['newrings']=np.where(df['Rings']>10,1,0)


# In[29]:


x = df.drop(['age','Rings','Sex','newrings'],axis=1)
y = df['newrings']


# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# In[31]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
pred
accuracy_score(y_test,pred)


# In[32]:


x= df.drop(['age','Rings','Sex'],axis=1)
y = df['Rings']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
from sklearn import svm
sv = svm.SVC(kernel='linear',C=1,gamma=1)
sv.fit(x_train,y_train)
pred = sv.predict(x_test)
pred
accuracy_score(y_test,pred)


# In[37]:


x= df.drop(['age','Rings','Sex'],axis=1)
y = df['Rings']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
from sklearn import svm
sv = svm.SVC(kernel='rbf',C=1,gamma=100)
sv.fit(x_train,y_train)
pred = sv.predict(x_test)
pred
accuracy_score(y_test,pred)


# In[38]:


new_df=df.copy()
new_df['newRings_1'] = np.where(df['Rings'] <= 8,1,0)
new_df['newRings_2'] = np.where(((df['Rings'] > 8) & (df['Rings'] <= 10)), 2,0)
new_df['newRings_3'] = np.where(df['Rings'] > 10,3,0)


# In[39]:


new_df['newRings'] = new_df['newRings_1'] + new_df['newRings_2'] + new_df['newRings_3']


# In[47]:


x = new_df.drop(['Rings','age','Sex','newRings_1','newRings_2','newRings_3'], axis = 1)
y = new_df['newRings']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
from sklearn import svm
sv = svm.SVC(kernel='rbf',C=1,gamma=100)
sv.fit(x_train,y_train)
pred = sv.predict(x_test)
pred
accuracy_score(y_test,pred)


# In[50]:


from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='rbf',C=1,random_state=100)
scores = cross_val_score(clf,x,y,cv=5)


# In[51]:


scores


# In[52]:


print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[53]:


#so SVM is giving best accuracy 


# In[ ]:




