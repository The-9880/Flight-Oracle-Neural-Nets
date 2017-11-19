import keras
import numpy as np

model = keras.models.load_model('predictTime.h5')

#   alright listen, this file takes 5 inputs: the 4 regular ones
#   The fifth input is the multiplicative combination of: wind, precipitation, and average temperature.

#   EXCEPT THE LAST ONE, the inputs have to be normalized by using the same subtractions/divisions as the last time.
#   Seriously, the fifth input is an absurd value but eh, works.

testData = [-0.628228333,2.671389566,-0.201493861,0.493882335,4719.715]
testData = np.reshape(testData, (1,5))

prediction = model.predict(testData, verbose=0)
print("The minimum waiting time predicted is: ", int(prediction), " minutes.")