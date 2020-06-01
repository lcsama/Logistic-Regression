import argparse

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


def gradAscent(X, Y, lr, iteration):
    pass

def stocGradAscent(X, Y, lr, iteration):
    pass


def run_logistic(method,lr,iteration):
    X, Y = make_moons(200, noise=0.20,random_state=0)
    #add a colume
    X=np.insert(X, 0, 1, axis=1)
    #change the Y to a mat
    Y=Y.reshape(200,1)
    if method == 'GN':
        weights = Gauss_Newton(X, Y,iteration)
    else:
        pass
        # weights = gradAscent(X, Y)
        # weights = stocGradAscent(X, Y)
    plotBestFit(X, Y, weights)
    


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('solve_method', type=str, help='Name of slove method.GN = GaussNewton')
    parser.add_argument('--learning_rate', type=float, default = 0.001, help='The Learning Rate of GD.')
    parser.add_argument('--iteration', type=int, default = 6, help='The iteration times.')
    args = parser.parse_args()
    run_logistic(args.solve_method,args.learning_rate,args.iteration)

if __name__ == "__main__":
    main()