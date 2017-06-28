import numpy as np
import mrmr as mr

########################################################################################################################
# This function runs the MRMR classificaion on all of the features and then uses those features to runs machine learning
# models to determine the accuracies based on the features.
########################################################################################################################

def classify(featureData, target, numclasses, ratio, tnum):
    sampleNum, featureNum = np.shape(featureData)
    featureRanking = np.zeros((tnum, featureNum))

    for i in range(tnum):
        # Randomize Data
        permuteRows = np.random.permutation(sampleNum)
        trainSize = int(np.round(sampleNum*ratio, decimals=0))
        trainingData = featureData[permuteRows[1:trainSize], :]
        trainingTarget = target[permuteRows[1:trainSize]]
        testData = featureData[permuteRows[trainSize+1:], :]
        testTarget = target[permuteRows[trainSize+1:]]

        # Rank Features
        featureSelection = mr.mrmr(trainingData, trainingTarget, numclasses, featureNum)
        featureRanking[i, :] = featureSelection[:]

    return featureRanking