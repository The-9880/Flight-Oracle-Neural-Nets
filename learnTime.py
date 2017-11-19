import keras
import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.linear_model


features=['WeatherDelayed', 'AWND', 'PCRP', 'SNOW', 'AVGT', 'CMPND']
keyFeatures = ['AWND', 'PCRP', 'SNOW', 'AVGT', 'CMPND']

dataSet = pd.read_csv('refinedTime.csv', index_col=False, usecols=features, names=features)


trainFeatures = dataSet[keyFeatures].values
trainLabels = dataSet['WeatherDelayed'].values
trainFeatures = trainFeatures[1:2000]
trainLabels = trainLabels[1:2000]

testFeatures = dataSet[keyFeatures].values
testFeatures = testFeatures[2001:2937]
testLabels = dataSet['WeatherDelayed'].values
testLabels = testLabels[2001:2937]


#   Model creation occurs now.
model = keras.models.Sequential()
model.add(keras.layers.Dense(300, activation='relu', kernel_initializer='normal', input_dim=5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(500, activation='relu', kernel_initializer='normal'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='normal'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(1, activation='relu', kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainFeatures, trainLabels, batch_size=128, epochs=50, validation_data=(testFeatures,testLabels))

score = model.evaluate(testFeatures, testLabels, verbose=0)
print("Accuracy: ", score)

print(model.predict(np.reshape([-0.628228333,2.671389566,-0.201493861,0.493882335,1.051748592],(1,-1))))
print(model.predict(np.reshape([-0.006416295,-0.495890416,-0.201493861,0.493882335,-0.615306047],(1,-1))))


model.save('predictTimeExperimental.h5')