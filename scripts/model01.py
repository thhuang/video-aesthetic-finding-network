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

X = np.load('/app/data/output/videos/score.npy')
Y = np.zeros(X.shape[0])
Y[100:-100] = 1

#####################
## Data Preprocess ##
#####################


####################
## Initialization ##
####################



