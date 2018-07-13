
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt

N = 500

#creating 3 gaussian clouds
X1 = np.random.randn(N, 2) + np.array([0, -2])
X2 = np.random.randn(N, 2) + np.array([2, 2])
X3 = np.random.randn(N, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

Y = np.array([0]*N + [1]*N + [2]*N)

plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()


# In[15]:


D = 2
M = 3
K = 3

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total


# In[16]:


P_I_given_X = forward(X, W1, b1, W2, b2)
P = np.argmax(P_I_given_X, axis=1)

assert(len(P) == len(Y))

print("Classification rate for randomly initialized weights:", classification_rate(Y, P))

