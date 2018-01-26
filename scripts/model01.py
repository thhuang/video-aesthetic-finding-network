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
num_time_steps = 10
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

# Define hyper-parameters
learning_rate = 1e-3
epochs = 800
batch_size = 2

# Define model
model = models.Sequential()
model.add(layers.LSTM(512, input_shape=(X.shape[1], X.shape[2]), activation='tanh', return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(512, activation='tanh'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(Y.shape[1], activation='softmax'))
model.summary()

# Define optimizer
adam = optimizers.Adam(learning_rate)
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set check point
filepath = '../data/output/practice/weights/weights-improvement={epoch:02d}-{loss:4f}.hdf5'
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