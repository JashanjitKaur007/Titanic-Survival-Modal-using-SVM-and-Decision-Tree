#!/usr/bin/env python
# coding: utf-8

# Importing Basic Libraries : -

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # LOADING THE DATASET

# In[2]:


file_path = r"C:\Users\ASUS\Downloads\titanic\train.csv"
#df is the train dataset
train = pd.read_csv(file_path)
train


# In[3]:


train.head()


# In[4]:


train.describe()


# In[5]:


train.info()


# # Data Analysis 

# In[152]:


# categorical attributes 
sns.countplot(x = 'Survived' , data = train , hue='Survived')


# In[148]:


sns.countplot(x = 'Sex' , data = train , hue='Sex')


# # DATA PRE-PROCESSSING 
# 
# --to manipulate data

# In[30]:


train_len = len(train)
df = pd.concat([train , test] , axis = 0)
df = df.reset_index(drop = True)
df.head()


# In[31]:


df.tail()


# In[32]:


df.isnull().sum()


# In[33]:


df = df.drop(columns=['Cabin'] , axis = 1)


# In[34]:


df['Age'].mean()


# In[35]:


# fill missing values using mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())


# In[36]:


df['Embarked'].mode()
# returns the mode value of each column.


# In[37]:


# fill missing values using mode of the categorical column
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


# TRAIN-TEST SPLIT
# --

# In[101]:


train = df.iloc[: train_len,:]
test = df.iloc[train_len : , :]


# In[102]:


train.head()


# In[103]:


test.head()


# In[116]:


# input split
X = train.drop(columns=['Survived'], axis=1)
y = train['Survived']
X.head()


# # MODEL TRAINING 

# In[117]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[130]:


df.dropna(subset=['Survived'],inplace = True)
df.dropna(inplace=True)


# In[131]:


X = df.drop(columns=['Survived'])
y = df['Survived']


# In[132]:


# Example:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[133]:


print(X.shape , X_train.shape , X_test.shape , y_train.shape)


# # MODEL TRAINING : - 

# # SVM(SUPPORT VECTOR MACHINE)

# In[141]:


from sklearn.svm import SVC


# In[142]:


categorical_features = X.select_dtypes(include=['object']).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)


# In[143]:


model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC())  # Using SVC (Support Vector Classifier) for SVM
])


# In[145]:


model.fit(X_train, y_train)


# ## Evaluating the Accuracy of the DTree Modal 

# In[146]:


accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)


# In[ ]:


# Thus the accuracy of the DT Modal is 0.53 out of 1


# ## DECISION TREE

# In[164]:


from sklearn.tree import DecisionTreeClassifier


# In[168]:


ategorical_features = X.select_dtypes(include=['object']).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)


# In[169]:


model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())  # Using Decision Tree Classifier
])


# In[170]:


model.fit(X_train, y_train)


# ## Evaluating the Accuracy of the DTree Modal 

# In[171]:


accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)


# In[ ]:


# Thus the accuracy of the SVM Modal is 0.73 out of 1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




