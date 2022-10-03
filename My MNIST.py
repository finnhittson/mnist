#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm.notebook import trange, tqdm

mnist = fetch_openml('mnist_784', version = 1)
mnist.keys()


# In[2]:


from sklearn.utils import shuffle
X, y = mnist['data'], mnist['target']
X = np.array(X)
y = np.array(y)
for i in range(len(y)):
    y[i] = int(y[i])
X, y = shuffle(X, y, random_state=datetime.now().microsecond)
m, n = X.shape


# In[3]:


# split data into test and training
y_test = y[0:1000]
X_test = X[0:1000]
X_test = X_test.T / 255

y_train = y[1000:]
X_train = X[1000:]
X_train = X_train.T / 255


# In[4]:


def initialize_parameters():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    
    return W1, b1, W2, b2


# In[5]:


def encoder(y):
    encoded = []
    for true_value in y:
        encoded.append([0] * int(true_value) + [1] + [0] * (9 - int(true_value)))
    return np.array(encoded).T


# In[6]:


def relu(Z):
    return np.maximum(Z, 0)


# In[7]:


def relu_p(Z):
    return Z > 0


# In[8]:


def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))


# In[9]:


def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    
    return Z1, A1, Z2, A2


# In[10]:


def backward_propagation(A1, A2, W2, Z1, Z2, X, y, m):
    dZ2 = A2 - encoder(y)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = np.expand_dims(1 / m * np.sum(dZ2, axis=1), axis = 1)
    
    dZ1 = W2.dot(dZ2) * relu_p(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = np.expand_dims(1 / m * np.sum(dZ1, axis=1), axis = 1)
    
    return dW1, db1, dW2, db2


# In[11]:


def update_parameters(W1, dW1, b1, db1, W2, dW2, b2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    
    return W1, b1, W2, b2


# In[123]:


def get_predictions(A):
    return np.argmax(A.T, 1)

def get_accuracy(predictions, y):
    #print(predictions, y)
    return np.sum(predictions == y) / y.size

def round_number(a):
    accuracy = str(round(a, 3))
    if len(accuracy) == 4 and accuracy[-1] != 0:
        accuracy += "0"
    return accuracy


# In[124]:


def gradient_descent(X, y, iterations, alpha, m):
    W1, b1, W2, b2 = initialize_parameters()
    t = trange(iterations)
    for i in t:
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(A1, A2, W2, Z1, Z2, X, y, m)
        W1, b1, W2, b2 = update_parameters(W1, dW1, b1, db1, W2, dW2, b2, db2, alpha)
        accuracy = round_number(get_accuracy(get_predictions(A2), y))
        t.set_description(accuracy, refresh=True)
    return W1, b1, W2, b2


# In[125]:


for i in range(len(y)):
    y[i] = int(y[i])
X, y = shuffle(X, y, random_state=datetime.now().microsecond)
m, n = X.shape

y_test = y[0:1000]
X_test = X[0:1000]
X_test = X_test.T / 255

y_train = y[1000:]
X_train = X[1000:]
X_train = X_train.T / 255

W1, b1, W2, b2 = gradient_descent(X_train, y_train, 500, 0.1, m)


# In[126]:


def make_predictions(W1, b1, W2, b2, X):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_predictions(index, W1, b1, W2, b2, X_data, y_data):
    current_image = X_data[:, index].reshape(784,1)
    prediction = make_predictions(W1, b1, W2, b2, current_image)
    label = y_data[index]
    print('Prediction: {}'.format(prediction))
    print('Label: {}'.format(label))

    current_image = current_image.reshape(28, 28) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# In[128]:


print(X_test.shape)
test_predictions(0, W1, b1, W2, b2, X_test, y_test)
test_predictions(1, W1, b1, W2, b2, X_test, y_test)
test_predictions(2, W1, b1, W2, b2, X_test, y_test)
test_predictions(3, W1, b1, W2, b2, X_test, y_test)
test_predictions(4, W1, b1, W2, b2, X_test, y_test)


# In[ ]:




