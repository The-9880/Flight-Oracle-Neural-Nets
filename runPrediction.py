import keras
import numpy as np

model = keras.models.load_model('delayPrediction.h5')

testData = [-.062945, -.60659, -.201494, -1.228397]
testData = np.reshape(testData, (1,4))

prediction = model.predict(testData, verbose=0)
print("It is ", int(prediction*100), "% likely that a flight will be delayed or cancelled in these conditions.")
