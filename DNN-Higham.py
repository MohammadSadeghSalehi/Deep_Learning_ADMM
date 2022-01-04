import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from numpy.random.mtrand import randn
import time 
np.random.seed(0)
t= time.time()
def activate(x,W,b):
    #sigmoid activation function
    return 1/(1+np.exp(-(W*x+b)))

def cost(W2,W3,W4,b2,b3,b4):
    costvec = np.zeros(100)
    for i in range(100):
        x =np.matrix(trainx[:,i].reshape((2,1)))
        a2 = activate(x,W2,b2)
        a3 = activate(a2,W3,b3)
        a4 = activate(a3,W4,b4)
        costvec[i] = np.linalg.norm(np.matrix(trainlable[:,i].reshape((2,1))) - a4)
    
    costval = np.linalg.norm(costvec)**2
    #print(costval)
    return costval
#Data
#The Data is 100 random points in R^2 which are labled randomly
trainx = np.random.randn(2,100)
trainlable = np.zeros((2,100))
for i in range(100):
    if np.random.randn() < 0.5:
        trainlable[0][i] = 1
    else:
        trainlable[1][i] = 1

#test Data
testx = np.random.randn(2,20)
testlable = np.zeros((2,20))
for i in range(20):
    if np.random.randn() < 0.5:
        testlable[0][i] = 1
    else:
        testlable[1][i] = 1


W2 = np.matrix(0.5*np.random.randn(2,2))
W3 = np.matrix(0.5*np.random.randn(3,2))
W4 = np.matrix(0.5*np.random.randn(2,3))
b2 = np.matrix(0.5*np.random.randn(2).reshape(2,1))
b3 = np.matrix(0.5*np.random.randn(3).reshape(3,1))
b4 = np.matrix(0.5*np.random.randn(2).reshape(2,1))
# Forward and Back propagate
eta = 0.05
# learning rate
Niter = int(5*1e4)
trainingAccuracy =0
testAccuracy = 0
accur=[]
testAccure=[]

#number of SG iterations
savecost = np.zeros(Niter)# value of cost function at each iteration

for counter in range (Niter):
    k = np.random.randint(100)
    l = np.random.randint(20)
    # choose a training point at random
    x = trainx[:,k]
    tx = testx[:,l]
    x = np.matrix(x.reshape((2,1)))
    tx = np.matrix(tx.reshape((2,1)))
    # Forward pass
    a2 = activate(x,W2,b2)
    a3 = activate(a2,W3,b3)
    a4 = activate(a3,W4,b4)
    # Backward pass
    delta4 = np.multiply(np.multiply(a4,(np.ones((2,1))-a4)),(a4-np.matrix(trainlable[:,k].reshape((2,1)))))
    delta3 = np.multiply(np.multiply(a3,(np.ones((3,1))-a3)),(np.transpose(W4)*delta4))
    delta2 = np.multiply(np.multiply(a2,(np.ones((2,1))-a2)),(np.transpose(W3)*delta3))
    #Gradient step
    W2 = W2 - eta*delta2*np.transpose(x)
    W3 = W3 - eta*delta3*np.transpose(a2)
    W4 = W4 - eta*delta4*np.transpose(a3)
    b2 = b2 - eta*delta2
    b3 = b3 - eta*delta3
    b4 = b4 - eta*delta4

    temp = np.zeros((2,1))
    temp[np.argmax(a4)][0] = 1
    if np.sum(temp==np.matrix(trainlable[:,k].reshape((2,1))))==2:
        trainingAccuracy+=1

    accur.append(trainingAccuracy/Niter)
    ##test
    a2 = activate(tx,W2,b2)
    a3 = activate(a2,W3,b3)
    a4 = activate(a3,W4,b4)
    temp = np.zeros((2,1))
    temp[np.argmax(a4)][0] = 1
    if np.sum(temp==np.matrix(testlable[:,l].reshape((2,1))))==2:
        testAccuracy+=1

    testAccure.append(testAccuracy/Niter)
    #train cost
    newcost = cost(W2,W3,W4,b2,b3,b4)
    savecost[counter]= newcost

plot1 = plt.figure(1)
plt.plot(np.linspace(0,Niter,Niter),accur,'orange')
plt.xlabel("Number of iterations of stochastic gradient")
plt.ylabel("Training & test accuracy")
plt.plot(np.linspace(0,Niter,Niter),testAccure,'green')
plt.legend(['Training','Test'])
#plt.yscale("log")
plot2 = plt.figure(2)
plt.plot(np.linspace(0,Niter,Niter),savecost,'red')
plt.xlabel("Number of iterations of stochastic gradient")
plt.ylabel("Cost")
t=time.time()-t
print("Time: ", t)
plt.show()

