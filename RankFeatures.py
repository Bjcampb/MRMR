import numpy as np
import classify as cl

# Load Data
WAT = np.transpose(np.loadtxt('C:\\Users\\bjcampb\\Google Drive\\Peak_Manuscript\\Data\\WAT.csv', delimiter=','))
BAT = np.transpose(np.loadtxt('C:\\Users\\bjcampb\\Google Drive\\Peak_Manuscript\\Data\\BAT.csv', delimiter=','))
MUSCLE = np.transpose(np.loadtxt('C:\\Users\\bjcampb\\Google Drive\\Peak_Manuscript\\Data\\Muscle.csv', delimiter=','))
WAT_FF = np.expand_dims(
    np.transpose(np.loadtxt('C:\\Users\\bjcampb\\Google Drive\\Peak_Manuscript\\Data\\FF_WAT.csv', delimiter=',')),
    axis=1)
BAT_FF = np.expand_dims(
    np.transpose(np.loadtxt('C:\\Users\\bjcampb\\Google Drive\\Peak_Manuscript\\Data\\FF_BAT.csv', delimiter=',')),
    axis=1)
MUSCLE_FF = np.expand_dims(
    np.transpose(np.loadtxt('C:\\Users\\bjcampb\\Google Drive\\Peak_Manuscript\\Data\\FF_Muscle.csv', delimiter=',')),
    axis=1)

# Combine Fat Fraction to feature set
WAT_Feat = np.concatenate((WAT, WAT_FF), axis=1)
BAT_Feat = np.concatenate((BAT, BAT_FF), axis=1)
MUSCLE_Feat = np.concatenate((MUSCLE, MUSCLE_FF), axis=1)

# Create classes
WAT_Class = np.zeros((np.shape(WAT_Feat)[0], 1))
BAT_Class = np.zeros((np.shape(BAT_Feat)[0], 1)) + 1
MUSCLE_Class = np.zeros((np.shape(MUSCLE_Feat)[0], 1)) + 2

# Create Feature Matrix and Class matrix
feat_data = np.concatenate((WAT_Feat, BAT_Feat, MUSCLE_Feat))
target = np.concatenate((WAT_Class, BAT_Class, MUSCLE_Class))
target = np.squeeze(target.astype(int))

# Run classification
ratio = 1.0
tnum = 1
numclasses = 3

Ranking = cl.classify(feat_data, target, numclasses, ratio, tnum)