import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

#from tensorflow.contrib.keras import models
#from tensorflow.contrib.keras import layers
from keras import models
from keras import layers



######################
## Data Preparation ##
######################

# Input sequence for RNN-based network : [samples, time steps, features]
# convert the output patterns into a one hot encoding

#video_feature_dir = '/app/data/output/videos/tainan/bridge/bridge_VIRB_V0330045_edit_1_label_85.mp4.npy'
#video_feature_dir = '/app/data/output/videos/tainan_demo/AIlabs_Demo.mp4.npy'
video_feature_dir = '/app/data/output/videos/test/test01.mp4.npy'
X = np.load(video_feature_dir)
X  = np.expand_dims(X, axis=0)

num_samples, num_time_steps, num_features = X.shape


####################
## Initialization ##
####################

# Define model
model = models.Sequential()
model.add(layers.Bidirectional(layers.GRU(200, activation='tanh', return_sequences=True), input_shape=(num_time_steps, num_features)))
model.add(layers.Dropout(0.2))
model.add(layers.Bidirectional(layers.GRU(200, activation='tanh', return_sequences=True)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# Load model
model.load_weights('/app/data/output/videos/tainan/weights/weights-improvement=10-0.161847.hdf5')


################
## Evaluation ##
################

print('Evaluation started!')
start_time = datetime.now()
result = model.predict_proba(X)[0][:, 1]
result_threshold = result > 0.3  # TODO: threshold
end_time = datetime.now()
print(result)
if result_threshold[0] == 1:
    print('clip: {:.2f}'.format(0), end=', ')
    state = True
else:
    state = False
for i in range(len(result) - 1):
    if not result_threshold[i] == result_threshold[i + 1]:
        if not state:
            print('clip: {:.2f}'.format(i / 30), end=', ')
            state = True
        else:
            print('{:.2f}'.format(i / 30))
            state = False
if state:
    print('end')
print('Evaluation complete!')
print('Time taken: {}'.format(end_time - start_time))
plt.plot(np.arange(0, len(result)) / 30, result)
plt.savefig('/app/data/output/videos/result')