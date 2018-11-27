import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)


def loadData(file, delimeter):
    # type: (object, object) -> object
    data = np.loadtxt(file, delimiter=delimeter)
    print ('Dimensions:', data.shape)
    print(data[1:6, :])
    return data


def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1

    if axes == None:
        axes = plt.gca()
        axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
        axes.scatter(data[neg][:, 0], data[neg][:, 1], c='y', s=60, label=label_neg)
        axes.set_xlabel(label_x)
        axes.set_ylabel(label_y)
        axes.legend(frameon=True, fancybox=True)

data = loadData('F:\\MLcode\\logsitc-regreesion\\data1.txt', ',')
X = np.c_[np.ones((data.shape[0], 1)), data[:, 0:2]]
y = np.c_[data[:, 2]]
fig = plt.figure()
fig.add_subplot(1, 1, 1)
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')
plt.show()
