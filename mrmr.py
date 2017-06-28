import numpy as np
import ftest as ft
import euclideandistance as eu

#####################################################################################
# Find First Feature
#####################################################################################
def FirstFeature(trainingData, trainingTarget, numclasses, numfeatures):
    # Determines top ranked feature based only on the F-test
    # This is the start of the algorithm in the MRMR paper
    rank = []
    for i in range(numfeatures):
        ftest = ft.ftest(trainingData[:, i], trainingTarget, numclasses)
        rank.append(ftest)

    topFeature = np.argmax(rank)

    return topFeature

####################################################################################
# Criterion Function
####################################################################################
def CriterionFunction(trainingData, trainingTarget, numclasses, totalrank, featurelist):

    rank = []
    set_size = np.shape(totalrank)[0] + 1

    running_total_ftest = 0
    running_total_distance = 0

    # Determines ftest and distance for previously selected top features
    for i in totalrank:
        running_total_ftest += ft.ftest(trainingData[:, i], trainingTarget, numclasses)
        running_total_distance += eu.euclideandistance(trainingData[:, i])


    for j in featurelist:
        new_ftest = ft.ftest(trainingData[:, j], trainingTarget, numclasses)
        new_distance = eu.euclideandistance(trainingData[:, j])

        result = new_ftest * (set_size / new_distance)
        rank.append(result)

    next_feature_location = np.argmax(rank)

    return next_feature_location



#####################################################################################
# Main Function
#####################################################################################
def mrmr(trainingData, trainingTarget, numclasses, numfeatures):

    totalrank = [] # List that keeps track of feature ranks
    featurelist = list(range(numfeatures)) # List that keeps track of remaining features to test

    # Determine Top Feature
    topFeature  = FirstFeature(trainingData, trainingTarget, numclasses, numfeatures)
    totalrank.append(topFeature) # Places top ranked feature to the start of list
    featurelist.pop(topFeature) # Removes previously selected top ranked feature from feature list

    # Determine Remaining Features
    while len(featurelist) != 0:
        feature_index = CriterionFunction(trainingData, trainingTarget, numclasses, totalrank, featurelist)

        totalrank.append(featurelist[feature_index]) # Add next highest ranked feature to total rank
        featurelist.pop(feature_index) # Remove next highest ranked feature from feature list

        print(len(featurelist))

    return totalrank












