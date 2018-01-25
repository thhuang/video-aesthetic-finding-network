import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from datetime import datetime


######################
## Data Preparation ##
######################

raw_text = open('../data/input/practice/frost.txt').read().lower()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

#Number of characters in the dataset
num_chars = len(raw_text)
#Vocabulary size of the dataset
num_vocab = len(chars)
#Length in which the senetences will be broken
seq_length = 100

data_input = []
data_target = []
for i in range(0, num_chars - seq_length, 1):
    seq_in = raw_text[i: i + seq_length]
    seq_out = raw_text[i + seq_length]
    # Converting characters to integers using char_to_int dictionary
    data_input.append([char_to_int[c] for c in seq_in])
    data_target.append(char_to_int[seq_out])


#####################
## Data Preprocess ##
#####################

#Input sequence for LSTM network : [samples, time steps, features]
#rescale integers from 0-1 since LSTM uses sigmoid to squash the activation values in [0,1]
#convert the output patterns into a one hot encoding

num_samples = len(data_input)
num_time_steps = seq_length
num_features = 1  # char index

# Reshape X to be [samples, time steps, features]
X = np.reshape(data_input, (num_samples, num_time_steps, num_features))
# Normalize
X = X / num_vocab
# One-hot encoding
Y = np_utils.to_categorical(data_target)
print(X.shape)
print(Y.shape)


####################
## Initialization ##
####################

# Define hyper-parameters
learning_rate = 1e-3
epochs = 800
batch_size = 100

# Define model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.summary()

# Define optimizer
adam = Adam(learning_rate)
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set check point
filepath = '../data/output/practice/weights/weights-improvement={epoch:02d}-{loss:4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


##############
## Training ##
##############

print('Training started:')
start_time = datetime.now()
model.fit(X, Y,
          nb_epoch=epochs, batch_size=batch_size,
          shuffle=True, callbacks=callbacks_list,
          verbose=1)
end_time = datetime.now()
print('Training complete!')
print('Time taken: {}'.format(end_time - start_time))