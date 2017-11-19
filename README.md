# Flight-Oracle-Neural-Nets
This repo contains the machine learning work I've produced for a Hackathon project, dubbed Flight Oracle. The project's purpose is to
take as inputs weather conditions from a forecast and predict the likelihood that a flight meant to depart from an area affected
by such conditions may be delayed.

The csv files contained here are datasets my teammates collected and I've spent a good deal of time organizing/reordering and
reformatting to make for easier loading/use in my Neural Nets. The entire experience was extremely enriching with all the skill practice
and application I've done, and I was absurdly pleased to find that my classifier would output reasonable likelihoods for given inputs.

The files DelayCancelClassifier.py and learnTime.py are the ones that load data and train Neural Nets. DelayCancelClassifier trains
the Neural Network that predicts the likelihood of a flight being cancelled given a combination of weather conditions as inputs. We took
Wind speed, Precipitation, Snow, and Average Temperature, as these were the most readily available and usable datasets were difficult to
find. This file also generates the .h5 for its model.

learnTime.py was used to train regression to predict how long a person might have to wait given the event that their flight is likely to
be delayed. This one went less smoothly and I wound up using a Neural Net again, even though I didn't find it ideal for this regression
model.

The remaining Python files are used to load the generated models to make predictions - these are the scripts that were packaged with 
our website, along with the trained models/weights, to be used to make reasonable predictions based on a user's inputs. A user would be
expected to select their destination of choice and date of departure, so that we could use The Weather Network's API to pull forecast
information for that date and feed it into my NN to return the likelihood that the flight would be delayed, and by how much time.

Note: the predictions for time aren't sound, but at every observation I've found them to be a touch or a leap beneath the recorded
values in our training data - therefore, they could be considered 'safe' estimates, but I wouldn't bet on it always being that way.
