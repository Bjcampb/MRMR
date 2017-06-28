import numpy as np
import math

def euclideandistance(trainingData):



    sum = 0
    for i in range(np.shape(trainingData)[0]):
        for j in range(np.shape(trainingData)[0]):
            if i != j:
                distance = np.abs(math.sqrt((trainingData[i] - trainingData[j])**2))
                sum += distance
                #print(sum)


    return sum / 2