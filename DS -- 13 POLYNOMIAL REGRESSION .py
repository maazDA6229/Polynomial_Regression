#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\python first\100 python programs\polynomial_level_salary_dataset.csv")


# In[3]:


dataset.head(3)


# In[4]:


dataset.corr()


# In[5]:


plt.scatter(dataset["Level"],dataset["Salary"])
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# In[ ]:





# In[6]:


x = dataset[["Level"]]
y = dataset["Salary"]


# In[7]:


from sklearn.preprocessing import PolynomialFeatures


# In[8]:


pf = PolynomialFeatures(degree=4)
pf.fit(x)
x = pf.transform(x)


# In[ ]:





# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)


# In[ ]:





# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


lr = LinearRegression()
lr.fit(x_train,y_train)


# In[13]:


lr.score(x_test,y_test)*100


# In[14]:


# y = m1x1+m2x2^2+c


# In[15]:


lr.coef_


# In[16]:


lr.intercept_


# In[17]:


prd = lr.predict(x)


# In[18]:


plt.scatter(dataset["Level"],dataset["Salary"])
plt.plot(dataset["Level"],prd,c = "red")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend(["org","prd"])
plt.show()


# In[ ]:





# In[19]:


test = pf.transform([[45]])
test


# In[20]:


lr.predict(test)


# In[ ]:





# In[ ]:




