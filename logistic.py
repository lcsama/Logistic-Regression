import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons



#Draw the figure
def plotBestFit(dataArr, labelMat, weights):
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def Gauss_Newton(X, Y,iteration):
    m, n = np.shape(X)
    # initialise the weight
    weight = np.zeros((n, 1))
    for k in range(iteration):  # heavy on matrix operations
        hypothesis=sigmoid(X.dot(weight))
        # first_derivative
        first_derivative = X.T.dot(Y - hypothesis )
        #calculate the Hessian mat with Diagonal decomposition 
        diag=np.diagflat( hypothesis*( hypothesis-np.ones((m,1)) ) )
        H=np.dot(X.T.dot(diag),X)
        #update the weight
        weight-=np.linalg.solve(H,first_derivative)
    return weight

def Grad_Descent(X, Y, lr, iteration):
    m, n = np.shape(X)
    # initialise the weight
    weight = np.zeros((n, 1))
    for k in range(iteration):  # heavy on iteration
        hypothesis=sigmoid(X.dot(weight))
        # original it's a sum() operation like sum((Y[i]-hypothesis[i])x[i]), but we can change it to matrix mul
        weight += lr * (X.T @ (Y - hypothesis))  
    return weight

def stocGradAscent(X, Y, lr):
    m, n = np.shape(X)
    # initialise the weight
    weight = np.zeros(n)
    for k in range(m):  
        hypothesis=sigmoid(sum(X[k]*weight))
        weight += lr * (X[k] * (Y[k] - hypothesis))  
    return weight

def miniBatchGradAscent(X, Y, lr, iteration, batch_size):
    m, n = np.shape(X)
    # initialise the weight
    weight = np.zeros((n, 1))
    for k in range(iteration):
        # take m samples to GradAscent
        for i in range(0,m,batch_size):
            mX = X[i:i+batch_size]
            mY = Y[i:i+batch_size]
            hypothesis=sigmoid(mX.dot(weight))
            weight += lr * (mX.T @ (mY - hypothesis))  
    return weight


def run_logistic(method, lr, iteration, batch_size):
    start = time.time()
    X, Y = make_moons(200, noise=0.20,random_state=0)
    #add a colume as bias
    X=np.insert(X, 0, 1, axis=1)
    #change the Y to a mat
    Y=Y.reshape(200,1)
    if method == 'GN':
        weights = Gauss_Newton(X, Y, iteration)
    elif method == 'GD':
        weights = Grad_Descent(X, Y, lr, iteration)
    elif method == 'SGD':
        weights = stocGradAscent(X, Y, lr)
    elif method == 'MBGD':
        weights = miniBatchGradAscent(X, Y, lr, iteration, batch_size)
    else:
        print("Wrong method, try GN or GD or SGD or MBGD")
        return
    duration = time.time() - start
    print('{}:{}s'.format(method,duration))
    plotBestFit(X, Y, weights)
    


def main():
    parser = argparse.ArgumentParser(description='Run the program with the algorithm you want to use.')
    parser.add_argument('solve_method', type=str, help='Name of slove method.GN = GaussNewton')
    parser.add_argument('--learning_rate', type=float, default = 0.001, help='The Learning Rate of GD.')
    parser.add_argument('--iteration', type=int, default = 6, help='The iteration times.')
    parser.add_argument('--batch_size', type=int, default = 20, help='Batch size for batch GD.')
    args = parser.parse_args()
    
    run_logistic(args.solve_method,args.learning_rate,args.iteration, args.batch_size)
    

    

if __name__ == "__main__":
    main()