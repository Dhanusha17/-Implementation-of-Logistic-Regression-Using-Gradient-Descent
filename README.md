# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.

## Program:
/*
Program to implement the the Logistic Regression Using Gradient Descent.

Developed by: DHANUSHA K

RegisterNumber:212223040034
*/
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```
## Output:
## Array Value of x
![322345528-53ea9891-469f-4302-b2ab-89f5ae570930](https://github.com/Dhanusha17/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151549957/28a0bd75-9547-4b1a-aba0-9c410d1ff752)

## Array Value of y
![322345670-09a7d024-6494-441d-9483-47d4a87e2a46](https://github.com/Dhanusha17/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151549957/843acfc4-62e5-4216-9eb6-8b7fe866cc0b)

## Exam 1 - score graph
![322345809-f194cf9b-90f5-40ef-9e82-bf44cbae1fdf](https://github.com/Dhanusha17/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151549957/00cfa526-c33c-4145-9664-ef74058d99d9)

## Sigmoid function grapH
![322346012-41108f65-f4ad-45cb-868c-3f8178e6bd25](https://github.com/Dhanusha17/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151549957/75b76112-153e-471f-a872-c43723af028a)

## X_train_grad value
![322346073-eebcecce-8f51-4a50-956f-25f8b20f183c](https://github.com/Dhanusha17/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151549957/c81f93f3-58b8-4c56-8c39-24b2f34bd52b)

## Y_train_grad value
![322346270-302c3a41-65bb-441e-8191-41c2c7990be5](https://github.com/Dhanusha17/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151549957/62338a8f-1cef-4fc1-8a5a-1093ba8b1983)

## Print res.x
![322346391-7c6dbfca-4999-41da-99f9-cc693310faa8](https://github.com/Dhanusha17/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151549957/2d2c88f4-4879-4a12-951f-04c0720471c8)

## Decision boundary - graph for exam score
![322346502-9824c702-6751-4557-a856-c704625201de](https://github.com/Dhanusha17/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151549957/87099a7b-95ba-45b6-bf8e-df926f60ec9a)

## Proability value
![322346607-69062590-586b-42b6-8e2e-6857864c7dda](https://github.com/Dhanusha17/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151549957/c87a81c6-a1ff-4074-a736-ee1057e8b1a1)

## Prediction value of mean
![322346721-337bfe1a-0008-4817-8b99-6976c0b5dce8](https://github.com/Dhanusha17/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151549957/524e8c54-418f-4d37-bdab-cda3782a58c4)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

