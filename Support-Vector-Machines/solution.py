import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from data import inputs, targets, plotContour, plot

N = 0
C = 0
#This will find the vector alpha which minimizaes the function objective
#Within the boundries B and the constraints XC
def kernel_linear(x, y):
    return numpy.dot(x,y)

def kernel_poly(x, y, degree):
    return (numpy.dot(x, y) + 1) ** degree

def kernel_rbf(x, y):
    a = x - y
    sigma = 1
    return np.exp((np.sqrt(np.dot(a, a)) ** 2) / (2 * sigma ** 2))

def zerofun(a, t):  
    return numpy.dot(a, t)

def objective(alpha_):
    result = 0
    for i in range(N):
        for j in range(N):
            right_side = 0.5 * alpha[i] * alpha[j] * p[i][j] 
            left_side = -alpha[i]
            result += (right_side + left_side)
    return result

#Pi,j = titjK(⃗xi, ⃗xj )
def build_matrix(N, x, t, kernel_function, P):
    for i in range(N):
        for j in range(N):
            P[i][j] = t[i] * t[j] * kernel_function(x[i], x[j])
    return P

def learn(x, t):
    N = x.shape[0]
    P = [N][N]
    P = build_matrix(N, x, t, kernel_linear, P)

def indicator(alpha, T, X, b, kernel_function, x):
    numpy.sum(alpha * T * numpy.array([kernel_function(x, xi) for xi in X])) - b


if __name__ == "__main__":
    x, t = learn(inputs, targets)
    B = [(0, C) for b in range(N)]
    start = numpy.zeros(N)
    XC = {'type':'eq', 'fun':zerofun}
    ret = minimize(objective, start, bounds=B, constraints=XC)
    found = ret['success']
    alpha = ret['x']
    print('alpha: ', alpha)
    non_zero = numpy.argwhere(alpha > 10 ** -5).flatten()

    #the support vectors give non zero values
    alpha, X, T = alpha[non_zero], x[non_zero], t[non_zero]
    b = indicator(alpha, T, X, 0, kernel_linear, X[0]) - T[0]