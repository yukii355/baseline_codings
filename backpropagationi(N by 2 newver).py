
# -*- coding: utf-8 -*-

import random #library related to random

import numpy as np
import sys
import matplotlib
import matplotlib
#matplotlib.use("gtk")
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    return np.tanh(z)

'''

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T # ndim = 2

'''


'''
def softmax(a):
    c = np.max(a)
    exp_a = np.max(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

'''



def softmax(x):
    x = np.exp(x) #分子
    sum_x = np.sum(x) #分母
    return x/sum_x # y を返す



# 2乗和誤差
def mean_squared_error(y, t):
    # ニューラルネットワークの出力と教師データの各要素の差の2乗、の総和
    return 0.5 * np.sum((y - t) ** 2)



# 交差エントロピー誤差
def cross_entropy_error(y, t):
  delta = 1e-7 # マイナス無限大を発生させないように微小な値を追加する
  #print("y-->")
  #print(y)
  return -np.sum(t * np.log(y + delta))








N = 6 #no. of data
#b = 0.5 #this is bias but I determined this value appropriately
#b2 = 0.5
a = 1.0 #a = learning rate


#X = np.random.rand(N, 2) #get the float value from 0.0~1.0

X = np.array([[0.1, 0.1], [0.2, 0.2],[0.9, 0.8],[0.4, 0.4],[0.4, 0.4],[0.9, 0.9]])
#X = np.array([0.1, 0.1])

t = np.array([[0,0],[0,0],[1,1],[0,0],[0,0],[1,1]])
#t = np.array([0,1])

print(X)

hidden_size = 10

W1 = np.random.rand(2,hidden_size)
b = np.zeros(hidden_size)

print("W1-->")
print(W1)

#h = np.zeros(10)


W2 = np.random.rand(hidden_size,2)
b2 = np.zeros(2)


print("W2-->")
print(W2)

Y = np.zeros((1, 1))

K = np.zeros(2)


def forward(_x,_w,_b):  # what the  " _ " mean is an internal variables

    _h = sigmoid(np.dot(_x, _w) + _b)

    return _h


all_loss = np.zeros([1000,1])

for step in range(1000):
    #print(W1)
    #print(W2)
    #print(b)
    #print(b2)

    h = forward(X, W1, b)

    K = forward(h, W2, b2)

   
    #print("h_shape"+str(h.shape))

    #print("K_shape"+str(K.shape))
    #print("W2_shape"+str(W2.shape))
    #print("h_shape"+str(h.shape))
    #print(N)






    #    k2 = sigmoid(h[k][0]* W2[0][1] + h[k][1] * W2[1][1] + h[k][2] * W2[2][1] + b2)  # what is the value of b2?

    #print("K-->")
    #print(K)
    #print(k2.shape)





#if I imput K(both K1 and K2) to softmax function, using below codes.


    #S = softmax(K)
    #print("S-->")
    #print(S)
    #print(S.shape)





    #y2 = Q[:N ,1]
    #print('error')
    #print(S)
    #print("loss-->")
    print(step, mean_squared_error(K,t)) #this is a loss value
    #print(step, cross_entropy_error(S,t)) #this is a loss value
    #print(mean_squared_error(np.array(y2), np.array(t)))
    all_loss[step, 0] = mean_squared_error(K,t)


    db1 = np.mean((K - t) * K * (1 - K),axis = 0) #axis= 0 is an average of columns. If axis = 1, average in rows.
    print("db1-->")
    print(db1)

    sum = 0
    for i in range (len(h)):
        sum += np.kron(h.T[:, i], db1).reshape((hidden_size, 2)) #np.kron is the tensor product
    dW1 = sum / len(h)
    print("dW1-->")
    print(dW1)


    #print("B-->")
    #print(B)

    dh = np.dot(db1, W2.T)
    #print("dh-->")
    #print(dh)


    db2 = np.mean(dh* h *(1- h),axis = 0)
    print("db2-->")
    print(db2)


    sum2 = 0
    for j in range(len(X)):
        sum2 += np.kron(X.T[:, j], db2).reshape((2,hidden_size))
    dW2 = sum2 / len(X)
    print("dW2-->")
    print(dW2)

    #print("E-->")
    #print(E)


    #below is the updating the data of bias and weights
    b2 = b2 - a * db1
    W2 = W2 - a * dW1
    b = b - a * db2
    W1 = W1 - a * dW2





plt.interactive(False)
plt.plot(all_loss)
print("all_loss-->")
print(all_loss)
plt.ylabel('some numbers')
plt.xlabel('some numbers')
plt.grid()
plt.show()

h = forward(X, W1, b)

K = forward(h, W2, b2)


print('printing K after finish training--------------')
print(K.shape)
print(K)

'''

# if I imput each K1 and K2 to softmax function, using below codes. 

print('part 1 shdjkashdkasd')
K1 = K[:N, 0]  #put out the values from matrix K, from column 0 to N-1, from row only 0
K2 = K[:N, 1]  #put out the values from matrix K, from column 0 to N-1, from row only 1


print(K1)
print(K2)


K3 = softmax(K1)
print('bringing softmax')
print(K3)
softmax_sum1 = np.sum(softmax(K1))#numpyのsumで合計が出る。
print(softmax_sum1)


K4 = softmax(K2)
print(K4)
softmax_sum2 = np.sum(softmax(K2))#numpyのsumで合計が出る。
print(softmax_sum2)

'''







'''

using loss function and training 




'''
