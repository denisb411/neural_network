
# coding: utf-8

# In[2]:


import numpy as np

K = 5 #number of classes
a = np.random.randn(K)
a


# In[3]:


expa = np.exp(a)
expa #now all positives


# In[4]:


answer = expa / expa.sum()
answer #probabilites of a


# In[5]:


answer.sum()


# In[7]:


N = 100 #number of samples
A = np.random.randn(N, K) #100 samples and 5 classes


# In[13]:


expA = np.exp(A)
answer = expA / expA.sum(axis=1) #because is trying to divide 1d array by 2d array


# In[15]:


answer = expA / expA.sum(axis=1, keepdims=True)
answer #N x K matrix


# In[17]:


answer.sum(axis=1)

