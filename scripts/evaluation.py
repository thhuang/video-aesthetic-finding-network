import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.insert(0, '/app')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from utils.time_counters import sec2time

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
video_feature_dir = '/data/thhuang/video-aesthetic-finding-network_output/videos/test/VIRB_V0190019_edit_0_label_218.mp4.npy'
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
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# Load model
model.load_weights('/data/thhuang/video-aesthetic-finding-network_output/videos/weights/weights-improvement=106-0.019243.hdf5')


################
## Evaluation ##
################

print('\nEvaluating...')
start_time = datetime.now()
result = model.predict_proba(X)[0]
end_time = datetime.now()
result_0 = result[:, 0]
result_1 = result[:, 1]
result_2 = result[:, 2]

result_argmax = np.argmax(result, axis=1)
print(result_argmax)

# Find clips
video_clips = list()
if result_argmax[0] == 1:
    clip = [0]
    state = True
else:
    state = False
for i in range(len(result) - 1):
    if not result_argmax[i] == result_argmax[i + 1]:
        if state:
            clip.append(i / 30)
            video_clips.append(clip)
            state = False
        else:
            clip = [i / 30]
            state = True
if state:
    clip.append(i / 30)
    video_clips.append(clip)

# Find clips not too short
final_clips = list()
for c in video_clips:
    if c[1] - c[0] >= 4:  # TODO: threshold
        final_clips.append(c)
        print('clip', sec2time(c[0]), sec2time(c[1]))

# Plot result
print('Evaluation complete!')
print('Time taken: {}'.format(end_time - start_time))
t = np.arange(0, len(result)) / 30
plt.plot(t, result_0, t, result_1, t, result_2)
plt.legend(['padding', 'good', 'bad'])
plt.savefig('/data/thhuang/video-aesthetic-finding-network_output/videos/result/result')
plt.close()
plt.plot(t, result_argmax == 1)
plt.savefig('/data/thhuang/video-aesthetic-finding-network_output/videos/result/result_clips')
plt.close()
