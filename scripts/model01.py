import numpy as np

from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import callbacks
from tensorflow.contrib.keras import utils
from datetime import datetime


######################
## Data Preparation ##
######################

data_input = np.load('/app/data/output/videos/score.npy')
data_target = np.zeros(data_input.shape[0])
data_target[30:-30] = 1

#####################
## Data Preprocess ##
#####################

#Input sequence for LSTM network : [samples, time steps, features]
#rescale integers from 0-1 since LSTM uses sigmoid to squash the activation values in [0,1]
#convert the output patterns into a one hot encoding

num_samples = len(data_input)
num_time_steps = 100
num_features = 1000  # deep features from VFN


# Reshape X to be [samples, time steps, features]
X = np.reshape(data_input, (num_samples, num_time_steps, num_features))
# Normalize
X = X / 300  # TODO: find better methods
# One-hot encoding
Y = utils.to_categorical(data_target)
print(X.shape)
print(Y.shape)


####################
## Initialization ##
####################

