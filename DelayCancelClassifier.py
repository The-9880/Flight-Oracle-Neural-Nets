import keras
import pandas as pd
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
import _pickle as cPickle
import keras.utils
from sklearn.model_selection import StratifiedKFold
import numpy as np

import numpy as np

#   The purpose of this Python script is to train a Neural Network on data we've collated from sources on the Internet
#   The Neural Network is meant to be a classifier which will take as inputs given details of the weather
#   ie: details of the weather forecast for, say, the day of your flight next week
#   And will then output the likelihood of that flight being delayed or canceled based on those weather conditions

#   Currently, we're training the classifier solely on data from Dallas
#   This presents a bit of a bias in estimations, as weather conditions in Dallas' climate aren't representative
#   Of conditions in opposite climates elsewhere in the world
#   However, weather conditions are weather conditions, regardless of the natural climate of an area.
#   So it may work just fine.

#   On further reflection:
#   The bias introduced, actually, is that the climate in Dallas does not capture the full range of possible
#   Weather conditions, because some conditions that may be observable in other countries may not ever
#   Be observed in Dallas, so we're missing data on those conditions.

allFeatures = ['WeatherDelay', 'AWND', 'PCRP', 'SNOW', 'AVGT']
keyFeatures = ['AWND','PCRP','SNOW','AVGT']

data = pd.read_csv("normalizedSnow.csv", index_col=False, names=allFeatures)    #   No na-values expected from our assembly of dataset

trainFeatures = data[keyFeatures].values
trainLabels = data['WeatherDelay'].values

trainFeatures = trainFeatures[1:]
trainLabels = trainLabels[1:]

for x in range(0, len(trainLabels)):    #   Turning this into a one-hot array
    trainLabels[x] = int(trainLabels[x])
    if(trainLabels[x] > 0):
        trainLabels[x] = 1

model = keras.models.Sequential()
model.add(keras.layers.Dense(50, activation='tanh', kernel_initializer='normal', input_dim=4))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(25, activation='tanh', kernel_initializer='normal'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(5, activation='tanh', kernel_initializer='normal'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer='normal'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainFeatures, trainLabels, batch_size=50, epochs=100, verbose=0)
score = model.evaluate(trainFeatures, trainLabels, verbose=0)
print("Accuracy: ", score[1])

model.save('delayPredictionExperimental.h5')


#   --- This stuff works:
#test = [2.9,0,0,30.1,0]
#test = np.reshape(test, (1,5))
#predictions = model.predict(test, verbose=0)
#print(predictions)