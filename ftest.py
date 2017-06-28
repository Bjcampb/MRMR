import numpy as np

def ftest(trainingData, trainingTarget, numclasses):

    totalDataSize = np.shape(trainingData)[0]
    K = numclasses # Number of classes
    sizeClasses = np.bincount(trainingTarget) # size of kth Class
    totalMean = np.mean(trainingData) # Mean value of all dataset
    classMean = [np.mean(trainingData[:sizeClasses[0]]),
                np.mean(trainingData[sizeClasses[0]:(sizeClasses[0]+sizeClasses[1])]),
                np.mean(trainingData[sizeClasses[1]+sizeClasses[2]:])]

    kthSigma = [np.var(trainingData[:sizeClasses[0]]),
                np.var(trainingData[sizeClasses[0]:(sizeClasses[0]+sizeClasses[1])]),
                np.var(trainingData[sizeClasses[1]+sizeClasses[2]:])]



    numerator = 0
    for i in range(K):
        numerator += ((sizeClasses[i] * (classMean[i] - totalMean))/(K - 1))



    sigma = 0
    for i in range(K):
        sigma += (((sizeClasses[i] - 1)*(kthSigma[i]*2)) / (totalDataSize - K))

    return numerator / (sigma * 2)