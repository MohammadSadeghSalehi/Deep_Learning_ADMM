import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from subprocess import check_output

import os
import gzip
import numpy as np
import mnist_reader


def relu(x):
    return  np.maximum(0,x)

def weight_update(layer_output, activation_input, rho):
    
    w = np.matmul(layer_output , np.linalg.pinv(activation_input))
    return w

def lambda_update(zl, w, a_in, beta):
    
    mpt = np.matmul(w , a_in)
    lambd = beta* (zl - mpt)
    return lambd

def argminz(a, w, a_in, beta, gamma):
    
    m = np.matmul(w, a_in)
    sol1 = (gamma*a + beta*m)/(gamma + beta)
    sol2 = m

    z1 = np.zeros((len(a),30000))
    z2 = np.zeros((len(a),30000))
    z =  np.zeros((len(a),30000))

    z1[sol1>=0] = sol1[sol1>=0]
    
    z2[sol2<=0] = sol2[sol2<=0]

    fz_1 = gamma*(a - relu(z1))**2 + beta* ((z1-m)**2)
    fz_2 = gamma*(a - relu(z2))**2 + beta* ((z2-m)**2)

    index_z1 = (fz_1<=fz_2)
    index_z2 = (fz_2<fz_1)

    z[index_z1] = z1[index_z1]
    z[index_z2] = z2[index_z2]
    return z

def argminlastz(targets, eps, w, a_in, beta):
    
    m = np.matmul(w,a_in)
    z = (targets - eps + beta*m)/(1+beta)
    return z

def argminlast_ez(y, eps, m, beta):
    return (y - eps + beta*m)/(1+beta)

def argmin_ez( a, m, beta, gamma):
    sol1 = (gamma*a + beta*m)/(gamma + beta)
    sol2 = m
    
    z1 = 0
    z2 = 0

    if sol1>=0:
        z1 = sol1
    else:
        z1 = 0

    if sol2<=0:
        z2 = sol2
    else:
        z2 = 0

    fz_1 = gamma*(a - relu(z1))**2 + beta* ((z1-m)**2)
    fz_2 = gamma*(a - relu(z2))**2 + beta* ((z2-m)**2)
    if fz_1<=fz_2:
        result = z1
    else:
        result = z2
    return result

def activation_update(next_weight, next_layer_output, layer_nl_output, beta, gamma):

    layer_nl_output = relu(layer_nl_output)

    # activation inverse
    m1 = beta*np.matmul(np.transpose(next_weight) , next_weight)
    m2 = gamma * np.eye(len(m1))
    av = np.linalg.inv(m1 + m2)

    #% activation formulate
    m3 = beta * np.matmul(np.transpose(next_weight) , next_layer_output)
    m4 = gamma * layer_nl_output
    af = m3 + m4

    a = np.matmul(av,af)
    return a


X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
X_train = np.transpose(X_train)#.reshape(784,60000)
X_train =X_train[:,0:30000]

X_test = np.transpose(X_test)
temp=np.zeros((10,int(len(y_train))))
for i in range(int(len(y_train))):
    temp[y_train[i]][i]=1
y_train = temp
y_train = y_train[:,0:30000]
temp=np.zeros((10,int(len(y_test))))
for i in range(int(len(y_test))):
    temp[y_test[i]][i]=1
y_test = temp

n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 #  2nd layer number of features
n_input = 784    #MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits) 
n_batchsize = 30000


beta3  = 5
beta2  = 5
beta1  = 5

gama2  = 5
gama1  = 5

rho    = 0.0

maxIter = 50
numLayer = 3

#running on CPU
#gADMM_NN(dataX, dataY, nn)

#init variable
a0 = np.zeros((n_input, n_batchsize))

w1 = np.zeros((n_hidden_1, n_input))#256x784
w2 = np.zeros((n_hidden_2, n_hidden_1))#256x256
w3 = np.zeros((n_classes, n_hidden_2))#10x256

z1 = np.random.randn(n_hidden_1, n_batchsize)#256xn
a1 = np.random.randn(n_hidden_1, n_batchsize)#256xn
z2 = np.random.randn(n_hidden_2, n_batchsize)#256xn
a2 = np.random.randn(n_hidden_2, n_batchsize)#256xn
z3 = np.random.randn(n_classes , n_batchsize)#10xn

lambd = np.ones((n_classes, n_batchsize))#10xn

a0 = X_train
y_labels  = y_train

#warm start
warm = 1
if warm:
    for i in range(1,2):
        print('--warming--')
        print(i)      
        
        #layer 1
        w1  =  weight_update(z1, a0, rho)
        a1  =  activation_update(w2, z2, z1, beta2, gama1)
        z1  =  argminz(a1, w1, a0, beta1, gama1)

        #layer 2
        w2  =  weight_update(z2, a1, rho)
        a2  =  activation_update(w3, z3, z2, beta3, gama2)
        z2  =  argminz(a2, w2, a1, beta2, gama2)

        #layer 3
        w3  =  weight_update(z3, a2, rho)
        z3  =  argminlastz(y_labels, lambd, w3, a2, beta3)


iter_TrainLoss = []
loss_train =[]
iter_TestLoss = []
loss_test =[]
iter_TrainAccuracy = []
accracy_train =[]
iter_TestAccuracy = []
accracy_test = []
trainingAccuracy=0
for i in range(1,maxIter+1):
    print('----------------------------\n','--training--',i)
       
    #layer 1
    w1  =  weight_update(z1, a0, rho)
    a1  =  activation_update(w2, z2, z1, beta2, gama1)
    z1  =  argminz(a1, w1, a0, beta1, gama1)

    #layer 2
    w2  =  weight_update(z2, a1, rho)
    a2  =  activation_update(w3, z3, z2, beta3, gama2)
    z2  =  argminz(a2, w2, a1, beta2, gama2)

    #layer 3
    w3  =  weight_update(z3, a2, rho)
    z3  =  argminlastz(y_labels, lambd, w3, a2, beta3)
    lambd = lambda_update(z3, w3, a2, beta3)
    
    # Training data
    forward = np.matmul(w3,relu(np.matmul(w2,relu(np.matmul(w1,X_train)))))
    temp = (forward - y_train)**2
    loss_train .append(np.sum(temp)) 
    tp = np.zeros((10,30000))
    for k in range(30000):
        tp[np.argmax(forward, axis=0)[k]][k]=1
    #tp[][:] = 1
    # if np.mean(tp==y_train)
    #     trainingAccuracy+=1
    # [M1, I1] = max(y_labels)
    # [M2, I2] = max(forward)
    accracy_train .append(np.mean(np.mean(tp==y_train,axis=0),axis=0))

    print('----')

    #test data
    forward1 = np.matmul(w3,relu(np.matmul(w2,relu(np.matmul(w1,X_test)))))
    temp = (forward1 - y_test)**2
    loss_test .append(np.sum(temp)) 
    tp = np.zeros((10,10000))
    for k in range(10000):
        tp[np.argmax(forward1, axis=0)[k]][k]=1
    # [M11, I11] = max(y_test)
    # [M22, I22] = max(forward1)
    accracy_test .append(np.mean(np.mean(tp==y_test,axis=0),axis=0)) 
    
    #plot
    iter_TrainLoss .append(i) 
    iter_TrainAccuracy .append(i) 
    iter_TestLoss .append(i) 
    iter_TestAccuracy .append(i) 

    # figure(1);
    # hold on;
    # plot(iter_TrainLoss,loss_train,'r')
    # hold on;
    # plot(iter_TestLoss,'b');
    # xlabel('Iterations');
    # ylabel('loss');
    # legend('Train Loss','Test Loss')
    # drawnow();

    # figure(2);
    # hold on;
    # plot(iter_TrainAccuracy,'y');
    # hold on;
    # plot(iter_TestAccuracy,'g');
    # xlabel('Iterations');
    # ylabel('accuracy');
    # legend('Train Accuracy','Test Accuracy')
    # drawnow();

    plot1 = plt.figure(1)
    plt.plot(iter_TrainLoss,loss_train,'red')
    plt.xlabel("Number of iterations")
    plt.ylabel("Training & test loss")
    plt.plot(iter_TestLoss,loss_test,'blue')
    plt.legend(['Training Loss','Test Loss'])
    #plt.yscale("log")
    plot2 = plt.figure(2)
    plt.plot(iter_TrainAccuracy,accracy_train,'yellow')
    plt.xlabel("Number of iterations")
    plt.ylabel("Training & test accuracy")
    plt.plot(iter_TestAccuracy,accracy_test,'green')
    plt.legend(['Train Accuracy','Test Accuracy'])
    plt.show()