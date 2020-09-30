#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

os.chdir(r'...')


# In[2]:


import pandas as pd
import numpy as np

df = pd.read_csv('pulsar_stars.csv')


# In[3]:


#import seaborn as sns
#import matplotlib.pyplot as plt
#%matplotlib inline

#sns.pairplot(data=df,hue='target_class')


# In[4]:


X = df[[' Mean of the integrated profile',
       ' Standard deviation of the integrated profile',
       ' Excess kurtosis of the integrated profile',
       ' Skewness of the integrated profile', ' Mean of the DM-SNR curve',
       ' Standard deviation of the DM-SNR curve',
       ' Excess kurtosis of the DM-SNR curve', ' Skewness of the DM-SNR curve']].values

Y = df['target_class'].values
Y = Y.reshape(17898,1)


# In[5]:


from sklearn.model_selection import train_test_split

x,xt,y,yt = train_test_split(X,Y,test_size=0.25,random_state=40)
x,xt,y,yt = x.T,xt.T,y.T,yt.T


# In[6]:


def iniparams(layer_dims):
    
    params = {}
    for l in range(1,len(layer_dims)):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return params


# In[7]:


def sigmoid(Z):
    return(1/(1 + np.exp(-Z)), Z)


# In[8]:


def relu(Z):
    return(np.maximum(0,Z), Z)


# In[9]:


def linearfwd(W, A, b):
    
    Z = np.dot(W, A) + b
    linear_cache = (W, A, b)
    return Z, linear_cache


# In[10]:


def fwdactivation(W, A_prev, b, activation):
    
    if activation == 'sigmoid':
        Z, linear_cache = linearfwd(W, A_prev, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linearfwd(W, A_prev, b)
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)
    return A, cache


# In[11]:


def fwdmodel(x, params):
    
    caches = []
    L = len(params)//2
    A = x
    
    for l in range(1,L):
        A_prev = A
        A, cache = fwdactivation(params['W' + str(l)], A_prev, params['b' + str(l)], 'relu')
        caches.append(cache)
    AL, cache = fwdactivation(params['W'+str(L)], A, params['b' + str(L)], 'sigmoid')
    caches.append(cache)
    return AL,caches


# In[12]:


def J(AL,y):
    return(-np.sum(np.multiply(np.log(AL),y)+np.multiply(np.log(1-AL),1-y))/y.shape[1])


# In[13]:


def sigmoidbkwd(dA, cache):
    
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA*s*(1 - s)
    return dZ


# In[14]:


def relubkwd(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ


# In[15]:


def linearbkwd(dZ, cache):
    
    W, A_prev, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


# In[16]:


def bkwdactivation(dA, cache, activation):
    
    linear_cache, activation_cache = cache
    if activation == 'sigmoid':
        dZ = sigmoidbkwd(dA, activation_cache)
        dA_prev, dW, db = linearbkwd(dZ, linear_cache)
    if activation == 'relu':
        dZ = relubkwd(dA, activation_cache)
        dA_prev, dW, db = linearbkwd(dZ, linear_cache)
    return dA_prev, dW, db  


# In[17]:


def bkwdmodel(AL, y, cache):
    
    grads = {}
    L = len(cache)
    dAL = -(np.divide(y, AL) - np.divide(1 - y,1 - AL))
    current_cache = cache[L-1]
    
    grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = bkwdactivation(dAL, current_cache, 'sigmoid')
    
    for l in reversed(range(L - 1)):
        current_cache = cache[l]
        dA_prev, dW_temp, db_temp = bkwdactivation(grads['dA' + str(l + 1)], current_cache, 'relu')
        grads['dA' + str(l)] = dA_prev
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp
    
    return grads


# In[18]:


def optimize(params, grads, alpha):
    
    L = len(params)//2
    for l in range(1,L):
        params['W' + str(l)] = params['W' + str(l)] - alpha*grads['dW' + str(l)]
        params['b' + str(l)] = params['b' + str(l)] - alpha*grads['db' + str(l)]
    
    return params 


# In[26]:


def model(x, y, layer_dims, iters = 3000):
    
    costs = []
    params = iniparams(layer_dims)
    for i in range(1, iters):
        AL, caches = fwdmodel(x, params)
        cost = J(AL, y)
        grads = bkwdmodel(AL, y, caches)
        params = optimize(params, grads, 1.2)
        if i%100 == 0:
            print('The cost after', i, 'iterations is:', cost)
            costs.append(cost)
    return params, costs


# In[27]:


params, costs = model(x, y, [8, 5, 4, 3, 1])


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')


# In[29]:


def predict(x,params):

    AL, cache = fwdmodel(x,params)
    predictions = AL >= 0.5
    
    return predictions


# In[30]:


predictions = predict(xt,params)
print ('Accuracy: %d' % float((np.dot(yt,predictions.T) + np.dot(1-yt,1-predictions.T))/float(yt.size)*100)+ '%')
predictions = predictions.astype('int')

