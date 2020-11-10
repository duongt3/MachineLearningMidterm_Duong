'''

Q5 A Large Character Level LSTM

'''

import re
import numpy as np
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import tensorflow as tf
from keras.callbacks import CSVLogger

config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
# Load and clean a text file
def fClean_Load(filename):
    file = open(filename, encoding="utf8", errors='ignore') 
    #file = open(filename, 'rb')
    text = file.read()
    file.close()
    # Clean text
    words = re.findall(r'[a-z\.]+', text.lower())
    return ' '.join(words)

# load text / Complete novel "A Tale of Two Cities"
raw_text = fClean_Load('AToTC.txt')

# organize into sequences of characters

length = 20
lines = list()
for i in range(length, len(raw_text)):
    seq = raw_text[i-length:i+1]
    lines.append(seq)
print('Total lines: %d' % len(lines))

chars = sorted(list(set(''.join(lines))))
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
	encoded_seq = [mapping[char] for char in line]
	sequences.append(encoded_seq)
    
# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

sequences = np.array(sequences)
X1, y = sequences[:,:-1], sequences[:,-1]
temp = [to_categorical(x, num_classes=vocab_size) for x in X1]
X = np.array(temp)
y = to_categorical(y, num_classes=vocab_size)

##############################################################################
####################### Select and fit an appropriate model ##################
# 1) LSTM size, 2) Dropout, 3) epochs, and 4) batch_size #####################
##############################################################################

csv_logger = CSVLogger('Q5_perp.csv')

model = Sequential()
model.add(LSTM(100,input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X, y, epochs=10 , verbose=1, batch_size=64, callbacks=[csv_logger])

# Save and test using code from the Q4_Test
model.save('LargeLSTM_model_test.h5')
dump(mapping, open('LargeLSTM_mapping_test.pkl', 'wb'))
