import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loaddata(filename, delm='\t'):
    fr = open(filename)
    strarr = [line.strip().split(delm) for line in fr.readlines()]
    datarr = [map(float, line) for line in strarr]
    fr.close()
    return np.mat(datarr)


def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[: -(topNfeat+1): -1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def FigPLot(filepath):
    datamat = loaddata(filepath)
    lowdata, reconmat = pca(datamat, 1)
    print lowdata
    fig = plt.figure(dpi=128)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(datamat[:, 0].flatten().A[0], datamat[:, 1].flatten().A[0], marker='^', s=90, )
    ax.scatter(reconmat[:, 0].flatten().A[0], reconmat[:, 1].flatten().A[0], marker='.', s=50, c='red',)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show(fig)  


def main():
    FigPLot('F:\\MLcode\\machinelearninginaction-master\\machinelearninginaction-master\\Ch13\\testSet.txt')


if __name__ == '__main__':
    main()
