
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ecommerce_data.csv')
df.head(10)


# In[7]:


import numpy as np
import pandas as pd

def get_data():
	df = pd.read_csv('ecommerce_data.csv')
	data = df.as_matrix()

	X = data[:, :-1]
	Y = data[:, -1]

	X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std() #normalizing
	X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

	N, D = X.shape
	X2 = np.zeros((N, D+3))
	X2[:,0:(D-1)] = X[:,0:(D-1)]

	for n in range(N):
		t = int(X[n,D-1])
		X2[n,t+D-1] = 1

	Z = np.zeros((N,4))
	Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1

	assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)

	return X2, Y
	
def get_binary_data():
	X, Y = get_data()
	X2 = X[Y <= 1]
	Y2 = Y[Y <= 1]
	return X2, Y2


# In[29]:


X, Y = get_data()

M = 5
D = X.shape[1]
K = len(set(Y))
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    print(Z)
    return softmax(Z.dot(W2) + b2)

P_Y_given_X = forward(X, W1, b1, W2, b2)
predictions = np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y, P):
    return np.mean(Y == P)

print("Score:", classification_rate(Y, predictions))


# In[25]:


np.tanh(1) + 0.995


# In[30]:


forward(np.array([[1,2]]), np.array([[1,1], [1,0]]), 0, np.array([[0,1], [1,1]]), 0)


# In[33]:


np.exp(.761)/ (np.exp(.761) + np.exp(1.756))


# In[34]:


np.exp(1.765)/ (np.exp(.761) + np.exp(1.756))

