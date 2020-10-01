#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
os.chdir(r'C:\Users\HP\Desktop\PyNN\Deep Neural Networks\imp5')


# In[10]:


import pandas as pd
import numpy as np
df = pd.read_csv('data.csv')


# In[11]:


X = df[['race', 'gender', 'age', 'admission_type_id',
       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
       'num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1',
       'diag_2', 'diag_3', 'number_diagnoses']].values
Y = df['readmitted'].values
Y = Y.reshape(59557,1)


# In[12]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[13]:


from sklearn.model_selection import train_test_split
x, xt, y, yt = train_test_split(X, Y, test_size = 0.25, random_state = 40)
x, xt, y, yt = x.T, xt.T, y.T, yt.T


# In[23]:


def iniparams(layer_dims):
    params = {}
    for l in range(1, len(layer_dims)):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1])*0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return params


# In[32]:


def sigmoid(Z):
    return 1/(1 + np.exp(-Z)), Z


# In[33]:


def relu(Z):
    return np.maximum(0, Z), Z


# In[34]:


def linearfwd(W, A, b):
    Z = np.dot(W, A) + b
    linear_cache = (W, A, b)
    return Z, linear_cache


# In[35]:


def fwdactivation(W, A_prev, b, activation):
    if activation == 'relu':
        Z, linear_cache = linearfwd(W, A_prev, b)
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        Z, linear_cache = linearfwd(W, A_prev, b)
        A, activation_cache = sigmoid(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


# In[36]:


def fwdmodel(x, params):
    caches = []
    L = len(params)//2
    A = x
    for l in range(1, L):
        A_prev = A
        A, cache = fwdactivation(params['W' + str(l)], A_prev, params['b' + str(l)], 'relu')
        caches.append(cache)
    AL, cache = fwdactivation(params['W' + str(L)], A, params['b' + str(L)], 'sigmoid')
    caches.append(cache)
    return AL, caches


# In[37]:


AL, caches = fwdmodel(x_t, params)


# In[39]:


def J(AL, y):
    return -np.sum(np.multiply(np.log(AL), y) + np.multiply(np.log(1 - AL), (1 - y)))/y.shape[1]


# In[41]:


def relubkwd(dA, cache):
    Z = cache
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ


# In[42]:


def sigmoidbkwd(dA, cache):
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA*s*(1 - s)
    return dZ


# In[43]:


def linearbkwd(dZ, cache):
    W, A_prev, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    dA_prev = np.dot(W.T, dZ)
    return dW, dA_prev, db


# In[44]:


def bkwdactivation(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relubkwd(dA, activation_cache)
        dW, dA_prev, db = linearbkwd(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoidbkwd(dA, activation_cache)
        dW, dA_prev, db = linearbkwd(dZ, linear_cache)
    return dW, dA_prev, db


# In[17]:


def bkwdmodel(AL, y, cache):
    grads = {}
    L = len(cache)
    dAL = -(np.divide(y, AL) - np.divide(1 - y,1 - AL))
    current_cache = cache[L - 1]
    grads['dW' + str(L)], grads['dA' + str(L - 1)], grads['db' + str(L)] = bkwdactivation(dAL, current_cache, 'sigmoid')
    for l in reversed(range(L - 1)):
        current_cache = cache[l]
        grads['dW' + str(l + 1)], grads['dA' + str(l)], grads['db' + str(l + 1)] = bkwdactivation(grads['dA' + str(l + 1)], current_cache, 'relu')
    return grads


# In[18]:


def optimize(params, grads, alpha):
    L = len(params)//2
    for l in range(1, L + 1):
        params['W' + str(l)] = params['W' + str(l)] - alpha*grads['dW' + str(l)]
        params['b' + str(l)] = params['b' + str(l)] - alpha*grads['db' + str(l)]
    return params


# In[19]:


def model(x, y, layer_dims, iters = 3000):
    costs = []
    params = iniparams(layer_dims)
    for i in range(1, iters):
        AL, cache = fwdmodel(x, params)
        cost = J(AL, y)
        grads = bkwdmodel(AL, y, cache)
        params = optimize(params, grads, 1.2)
        if i%100 == 0:
            print('The cost after', i, 'iterations is', cost)
            costs.append(cost)
    return costs, params


# In[20]:


costs, params = model(x, y, [17, 5, 1])


# In[21]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.title('Iterations Vs Cost')
plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')


# In[22]:


def predict(x,params):
    AL, cache = fwdmodel(xt,params)
    predictions = AL >= 0.5
    return AL,predictions


# In[23]:


AL, predictions = predict(xt,params)
print ('Accuracy: %d' % float((np.dot(yt,predictions.T) + np.dot(1-yt,1-predictions.T))/float(yt.size)*100)+ '%')
predictions = predictions.astype('int')

