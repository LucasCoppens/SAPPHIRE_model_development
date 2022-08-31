from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM,Bidirectional
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def build_fully_connected_NN(seq_len):
    model = Sequential()
    model.add(Dense(30, activation='relu',input_shape=(seq_len, 4)))
    model.add(Dense(10, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=binary_crossentropy,
                  optimizer='adam',
                  metrics=[sensitivity, specificity])
    return model

def build_CNN(seq_len):
    model = Sequential()
    model.add(Conv1D(activation="relu", filters=5, kernel_size=6,
                     strides=1,
                     padding="valid",
                     input_shape=(seq_len, 4)))
    model.add(MaxPooling1D(pool_size=5, strides=1))
    model.add(Dense(15, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=binary_crossentropy,
                  optimizer='adam',
                  metrics=[sensitivity, specificity])
    return model

def build_LSTM(seq_len):
    model = Sequential()
    model.add(Bidirectional(LSTM(10, return_sequences=True), "sum"))
    model.add(Dense(10, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=binary_crossentropy,
                  optimizer='adam',
                  metrics=[sensitivity, specificity])
    return model

def build_CNN_LSTM(seq_len):
    model = Sequential()
    model.add(Conv1D(activation="relu", filters=5, kernel_size=8,
                     strides=1,
                     padding="valid",
                     input_shape=(seq_len, 4)))
    model.add(MaxPooling1D(pool_size=3, strides=1))
    model.add(Bidirectional(LSTM(10, return_sequences=True), "sum"))
    model.add(Dense(10, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=binary_crossentropy,
                  optimizer='adam',
                  metrics=[sensitivity, specificity])
    return model

def build_LSTM_CNN_LSTM(seq_len):
    model = Sequential()
    model.add(Bidirectional(LSTM(10, return_sequences=True), "sum"))
    model.add(Conv1D(activation="relu", filters=5, kernel_size=8,
                     strides=1,
                     padding="valid",
                     input_shape=(seq_len, 4)))
    model.add(MaxPooling1D(pool_size=3, strides=1))
    model.add(Bidirectional(LSTM(10, return_sequences=True), "sum"))
    model.add(Dense(10, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=binary_crossentropy,
                  optimizer='adam',
                  metrics=[sensitivity, specificity])
    return model
