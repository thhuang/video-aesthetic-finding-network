import os
import numpy as np

from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import callbacks
from tensorflow.contrib.keras import utils
from tensorflow.contrib.keras import preprocessing
from sklearn.utils import shuffle
from datetime import datetime


######################
## Data Preparation ##
######################

#videos_features_dir = '/app/data/output/videos/tainan/boat'
#videos_features_dir = '/data/thhuang/video-aesthetic-finding-network_output/videos/tainan/boat'
videos_features_dir = '/data/thhuang/video-aesthetic-finding-network_output/videos/good'
video_names = os.listdir(videos_features_dir)
num_videos = len(video_names)

data_good_input = list()
data_good_target = list()
for video_name in video_names:
    print('Loading {}'.format(video_name))
    video_feature_dir = os.path.join(videos_features_dir, video_name)
    video_feature = np.load(video_feature_dir)
    video_target = np.ones(video_feature.shape[0])
    data_good_input.append(video_feature)
    data_good_target.append(video_target)

###############################################################################################

#videos_features_dir = '/app/data/output/videos/tainan/bridge'
#videos_features_dir = '/data/thhuang/video-aesthetic-finding-network_output/videos/tainan/bridge'
videos_features_dir = '/data/thhuang/video-aesthetic-finding-network_output/videos/bad'
video_names = os.listdir(videos_features_dir)
num_videos = len(video_names)

data_bad_input = list()
data_bad_target = list()
for video_name in video_names:
    print('Loading {}'.format(video_name))
    video_feature_dir = os.path.join(videos_features_dir, video_name)
    video_feature = np.load(video_feature_dir)
    video_target = np.ones(video_feature.shape[0]) * 2
    data_bad_input.append(video_feature)
    data_bad_target.append(video_target)


#####################
## Data Preprocess ##
#####################

# Input sequence for RNN-based network : [samples, time steps, features]
# Convert the output patterns into a one-hot encoding
random_seed = 123
np.random.seed(random_seed)
data_bad_input_clips = list()
data_bad_target_clips = list()
for video, target in zip(data_bad_input, data_bad_target):
    i = 0
    length = np.random.randint(210, 360)
    while i+length < len(video):
        data_bad_input_clips.append(video[i: i+length, :])
        data_bad_target_clips.append(target[i: i+length])
        i += length
        length = np.random.randint(210, 360)


# Balance data size
data_bad_input_clips, data_bad_target_clips = shuffle(data_bad_input_clips, data_bad_target_clips, random_state=random_seed)
data_input_clips = data_good_input + data_bad_input_clips[:len(data_good_input)]
data_target_clips = data_good_target + data_bad_target_clips[:len(data_good_target)]

# Padding
X  = np.concatenate([preprocessing.sequence.pad_sequences(data_input_clips, maxlen=360, padding='post', truncating='post'),
                     preprocessing.sequence.pad_sequences(data_input_clips, maxlen=360, padding='pre', truncating='pre')], axis=0)
X  = [np.expand_dims(e, axis=0) for e in X]
X  = np.concatenate(X, axis=0)

Y  = np.concatenate([preprocessing.sequence.pad_sequences(data_target_clips, maxlen=360, padding='post', truncating='post'),
                     preprocessing.sequence.pad_sequences(data_target_clips, maxlen=360, padding='pre', truncating='pre')], axis=0)
Y = [np.expand_dims(e, axis=0) for e in Y]
Y = np.concatenate(Y, axis=0)

# One-hot encoding
# 0: padding
# 1: good
# 2: bad
Y = utils.to_categorical(Y, num_classes=3)

X, Y = shuffle(X, Y, random_state=random_seed)

print('X:', X.shape)
print('Y:', Y.shape)
good_count = 0
bad_count = 0
for i, y in enumerate(Y):
    if y[0, 1] or y[-1, 1] == 1:
        good_count += 1
    if y[0, 2] or y[-1, 2] == 1:
        bad_count += 1
print('Good:Bad = {}:{}'.format(good_count, bad_count))

num_samples, num_time_steps, num_features = X.shape
num_categories = Y.shape[2]


####################
## Initialization ##
####################

# Define hyper-parameters
learning_rate = 1e-4
epochs = 3
batch_size = 10

# Define model
model = models.Sequential()
model.add(layers.Bidirectional(layers.GRU(200, activation='tanh', return_sequences=True), input_shape=(num_time_steps, num_features)))
model.add(layers.Dropout(0.2))
model.add(layers.Bidirectional(layers.GRU(200, activation='tanh', return_sequences=True)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(Y.shape[2], activation='softmax'))
model.summary()

# Define optimizer
adam = optimizers.Adam(learning_rate)
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['acc'])

# Set check point
#filepath = '../data/output/videos/tainan/weights/weights-improvement={epoch:02d}-{loss:4f}.hdf5'
filepath = '/data/thhuang/video-aesthetic-finding-network_output/videos/weights/weights-improvement=03-0.102548.hdf5'
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

