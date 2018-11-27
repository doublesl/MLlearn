import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)


def warmUpExercise():
    return (np.identity(5))


data = np.loadtxt('F:\\MLcode\\Linear-regression\\linear_regression_data1.txt', delimiter=',')
X = np.c_[np.ones(data.shape[0]), data[:, 0]]
y = np.c_[data[:, 1]]

fig = plt.figure(1, (60, 80))
fig.add_subplot(2, 2, 1)
plt.scatter(X[:, 1], y, s=30, c='r', marker='x', linewidths=1)
plt.xlim(4, 24)

plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')


def ComputeCost(X, y, theta=np.zeros((2, 1))):
    m = y.size
    J = 0
    h = np.dot(X, theta)
    J = 1.0 / (2 * m) * np.sum(np.square(h - y))
    return J


def GradientDescent(X, y, theta=np.zeros((2, 1)), alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)
    for i in np.arange(num_iters):
        h = np.dot(X, theta)
        theta = theta - alpha*(1.0/m)*np.dot(X.T, h-y)
        J_history[i] = ComputeCost(X, y, theta)
    return (J_history, theta)


Cost_J, theta = GradientDescent(X, y)
print('theta: ', theta)
fig.add_subplot(2, 2, 2)
plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations')


xx = np.arange(5, 23)
yy = theta[0]+theta[1]*xx
fig.add_subplot(2, 2, 3)
plt.scatter(X[:, 1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(xx, yy, label='Linear regression (Gradient descent)')


regr = LinearRegression()
regr.fit(X[:, 1].reshape(-1, 1), y.ravel())
print y.ravel()

plt.plot(xx, regr.intercept_+regr.coef_*xx, label='Linear regression (Scikit-learn GLM)')

plt.xlim(4, 24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(loc=4)
plt.show(fig)
