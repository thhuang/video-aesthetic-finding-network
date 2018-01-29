import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from keras import utils
from keras import preprocessing
from sklearn.utils import shuffle
from datetime import datetime
from tqdm import tqdm


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
seed = 123
np.random.seed(seed)

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
data_bad_input_clips, data_bad_target_clips = shuffle(data_bad_input_clips, data_bad_target_clips, random_state=seed)
data_input_clips = data_good_input + data_bad_input_clips[:len(data_good_input)]
data_target_clips = data_good_target + data_bad_target_clips[:len(data_good_target)]

# Padding
X_clips  = np.concatenate([preprocessing.sequence.pad_sequences(data_input_clips, maxlen=360, padding='post', truncating='post'),
                           preprocessing.sequence.pad_sequences(data_input_clips, maxlen=360, padding='pre', truncating='pre')], axis=0)
X_clips  = [np.expand_dims(e, axis=0) for e in X_clips]
X_clips  = np.concatenate(X_clips, axis=0)

Y_clips  = np.concatenate([preprocessing.sequence.pad_sequences(data_target_clips, maxlen=360, padding='post', truncating='post'),
                           preprocessing.sequence.pad_sequences(data_target_clips, maxlen=360, padding='pre', truncating='pre')], axis=0)
Y_clips = [np.expand_dims(e, axis=0) for e in Y_clips]
Y_clips = np.concatenate(Y_clips, axis=0)

# One-hot encoding (3 categories)
# 0: padding
# 1: good
# 2: bad
Y_clips = utils.to_categorical(Y_clips, num_classes=3)

X_clips, Y_clips = shuffle(X_clips, Y_clips, random_state=seed)
good_count = 0
bad_count = 0
for i, y in enumerate(Y_clips):
    if y[0, 1] or y[-1, 1] == 1:
        good_count += 1
    if y[0, 2] or y[-1, 2] == 1:
        bad_count += 1
print('Good:Bad = {}:{}'.format(good_count, bad_count))

print('Creating training data')
num_choices = 20
num_combinations = 10000
X_concat = list()
Y_concat = list()
for i in tqdm(range(num_combinations)):
    choices = np.random.choice(np.arange(0, X_clips.shape[0]), num_choices)
    X_concat.append(np.expand_dims(np.concatenate(X_clips[choices], axis=0), axis=0))
    Y_concat.append(np.expand_dims(np.concatenate(Y_clips[choices], axis=0), axis=0))

X = np.concatenate(X_concat, axis=0)
Y = np.concatenate(Y_concat, axis=0)

print('X:', X.shape)
print('Y:', Y.shape)

num_samples, num_time_steps, num_features = X.shape
num_categories = Y.shape[2]


####################
## Initialization ##
####################

# Define hyper-parameters
learning_rate = 1e-4
epochs = 200
batch_size = 100

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
              metrics=['accuracy'])

# Set check point
#filepath = '../data/output/videos/tainan/weights/weights-improvement={epoch:02d}-{loss:4f}.hdf5'
filepath = '/data/thhuang/video-aesthetic-finding-network_output/videos/weights/weights-improvement={epoch:02d}-{val_loss:4f}.hdf5'
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]


##############
## Training ##
##############

print('Training started:')
start_time = datetime.now()
history = model.fit(X, Y,
                    epochs=epochs, batch_size=batch_size,
                    validation_split=0.2, shuffle=True,
                    callbacks=callbacks_list, verbose=1)
end_time = datetime.now()
print('Training complete!')
print('Time taken: {}'.format(end_time - start_time))


###################
## Summarization ##
###################

# Summarize the history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/data/thhuang/video-aesthetic-finding-network_output/videos/result/accuracy')
plt.close()

# Summarize the history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/data/thhuang/video-aesthetic-finding-network_output/videos/result/loss')
plt.close()