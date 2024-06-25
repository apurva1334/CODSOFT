#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


data = 'fraudTrain.csv'
df=pd.read_csv(data)


# In[6]:


data = 'fraudTest.csv'
dt=pd.read_csv(data)


# In[7]:


df.shape


# In[8]:


dt.shape


# In[9]:


df.head()


# In[10]:


dt.head()


# In[43]:


col_names =['ct','trans_date_trans_time','cc_num','merchant','category','amt','first','last','gender','street','city','state','zip','lat','long','city_pop','job','dob','trans_num','unix_time','merch_lat','merch_long','is_fraud']
df.columns = col_names
dt.columns = col_names


# In[44]:


df['is_fraud'].value_counts()


# In[45]:


x_train = df.drop(['is_fraud'],axis=1)
y_train = df['is_fraud']


# In[46]:


x_train.head()


# In[47]:


y_train.head()


# In[48]:


dt['is_fraud'].value_counts()


# In[49]:


x_test = dt.drop(['is_fraud'],axis=1)
y_test = dt['is_fraud']


# In[50]:


x_test.head()


# In[51]:


y_test.head()


# In[52]:


df.shape


# In[53]:


dt.shape


# In[54]:


df.dtypes


# In[55]:


dt.dtypes


# In[56]:


df.isnull().sum()


# In[57]:



# In[58]:


from sklearn.preprocessing import OrdinalEncoder


# In[59]:

import category_encoders as ce


# In[81]:


encoder=ce.OrdinalEncoder(cols=['ct','trans_date_trans_time','cc_num','merchant','category','amt','first','last','gender','street','city','state','zip','lat','long','city_pop','job','dob','trans_num','unix_time','merch_lat','merch_long'])
x_train=encoder.fit_transform(x_train)
x_test=encoder.transform(x_test)


# In[63]:


x_train.shape


# In[64]:


y_train.shape


# In[65]:


x_test.shape


# In[66]:


y_test.shape


# In[67]:


x_train.head()


# In[68]:


x_test.head()


# In[69]:


from sklearn.tree import DecisionTreeClassifier


# In[108]:


#instantiate the classifier
fraud_tr= DecisionTreeClassifier(criterion = 'gini',max_depth=5,random_state=0)
# fit the model
fraud_tr.fit(x_train,y_train)


# In[109]:


from sklearn.metrics import accuracy_score


# In[110]:


y_pred_train = fraud_tr.predict(x_train)
print('training set accuracy score :{0:0.4f}'.format(accuracy_score(y_pred_train,y_train)))


# In[111]:


y_pred_test = fraud_tr.predict(x_test)
print('test set accuracy score :{0:0.4f}'.format(accuracy_score(y_pred_test,y_test)))


# In[112]:


from sklearn.tree import DecisionTreeClassifier,plot_tree


# In[113]:


def plot_decision_tree(fraud_obj,feature_names,class_names):
    plt.figure(figsize=(10,10))
    plot_tree(fraud_obj,filled=True , feature_names=feature_names,class_names=class_names,rounded=False)
    plt.show()


# In[114]:


d_tree= plot_decision_tree(fraud_tr,['ct','trans_date_trans_time','cc_num','merchant','category','amt','first','last','gender','street','city','state','zip','lat','long','city_pop','job','dob','trans_num','unix_time','merch_lat','merch_long'],['0','1'])


# In[115]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
print('Confusion matrix\n\n', cm)


# In[116]:


class_names = ['0','1']
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[117]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_test))


# In[ ]:




