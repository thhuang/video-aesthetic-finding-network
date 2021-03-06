import os
import numpy as np

from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import callbacks
from tensorflow.contrib.keras import utils
from tensorflow.contrib.keras import preprocessing
from datetime import datetime


######################
## Data Preparation ##
######################

#videos_features_dir = '/app/data/output/videos/tainan/boat'
videos_features_dir = '/data/thhuang/video-aesthetic-finding-network_output/videos/tainan/boat'
video_names = os.listdir(videos_features_dir)
num_videos = len(video_names)

data_input = list()
data_target = list()
for video_name in video_names:
    print('Loading {}'.format(video_name))
    video_feature_dir = os.path.join(videos_features_dir, video_name)
    video_feature = np.load(video_feature_dir)
    video_target = np.ones(video_feature.shape[0])
    data_input.append(video_feature)
    data_target.append(video_target)

######################################

#videos_features_dir = '/app/data/output/videos/tainan/bridge'
videos_features_dir = '/data/thhuang/video-aesthetic-finding-network_output/videos/tainan/bridge'
video_names = os.listdir(videos_features_dir)
num_videos = len(video_names)

for video_name in video_names:
    print('Loading {}'.format(video_name))
    video_feature_dir = os.path.join(videos_features_dir, video_name)
    video_feature = np.load(video_feature_dir)
    video_target = np.ones(video_feature.shape[0]) * 2
    data_input.append(video_feature)
    data_target.append(video_target)


#####################
## Data Preprocess ##
#####################

# Input sequence for RNN-based network : [samples, time steps, features]
# convert the output patterns into a one hot encoding

X  = preprocessing.sequence.pad_sequences(data_input, padding='post')
X  = [np.expand_dims(e, axis=0) for e in X]
X  = np.concatenate(X, axis=0)

Y = preprocessing.sequence.pad_sequences(data_target, padding='post')
Y = [np.expand_dims(e, axis=0) for e in Y]
Y = np.concatenate(Y, axis=0)

# One-hot encoding
# 0: padding
# 1: good
# 2: bad
Y = utils.to_categorical(Y, num_classes=3)

print(X.shape)
print(Y.shape)

num_samples, num_time_steps, num_features = X.shape
num_categories = Y.shape[2]


####################
## Initialization ##
####################

# Define hyper-parameters
learning_rate = 1e-4
epochs = 3
batch_size = 2

# Define model
model = models.Sequential()
model.add(layers.Bidirectional(layers.GRU(200, activation='tanh', return_sequences=True), input_shape=(num_time_steps, num_features)))
model.add(layers.Dropout(0.2))
model.add(layers.Bidirectional(layers.GRU(200, activation='tanh', return_sequences=True)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(Y.shape[2], activation='softmax'))
model.summary()

# Define optimizer
adam = optimizers.Adam(learning_rate)
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['acc'])

# Set check point
#filepath = '../data/output/videos/tainan/weights/weights-improvement={epoch:02d}-{loss:4f}.hdf5'
filepath = '/data/thhuang/video-aesthetic-finding-network_output/videos/tainan/weights/weights-improvement={epoch:02d}-{loss:4f}.hdf5'
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


##############
## Training ##
##############

print('Training started:')
start_time = datetime.now()
model.fit(X, Y,
          epochs=epochs, batch_size=batch_size,
          shuffle=True, callbacks=callbacks_list,
          verbose=1)
end_time = datetime.now()
print('Training complete!')
print('Time taken: {}'.format(end_time - start_time))

