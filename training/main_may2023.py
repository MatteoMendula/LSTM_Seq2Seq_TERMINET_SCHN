import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from keras.models import Sequential, save_model, load_model
from keras.layers import Bidirectional, LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_percentage_error
import os

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

models_path = "../saved_models/normal/may2023"

def create_sequence(dataset, target, window, future):
    x_sequence, y_sequence = [], []
    for index in range(len(dataset) - window - future):
        x_sequence.append(dataset[index: index + window])
        y_sequence.append(target[index + window: index + window + future])
    return (np.asarray(x_sequence), np.asarray(y_sequence))

def evaluate_predictions(predictions_seq, y_test_seq):
    MSE = []
    for pred in range(len(y_test)):
        mse = mean_squared_error(y_test_seq[pred], predictions_seq[pred])
        MSE.append(mse)
    return MSE

def find_max_error(predictions, y_test, mean_mse, std_mse):
    max_errors = 0
    for pred in range(len(y_test)):
        mse = mean_squared_error(y_test[pred], predictions[pred])
    if mse > mean_mse + std_mse:
        max_errors += 1
    return max_errors

# read dataset may2023
df_reshaped = pd.read_pickle("../data/20230319_RTU_Dataset_PPC-Lab/combined_may2023.pkl")  

# Normalizing the values
standard_scaler = preprocessing.StandardScaler()
print(df_reshaped.head())
scaled_df = standard_scaler.fit_transform(df_reshaped[['MEM_USAGE', 'CPU_USAGE', 'TEMP']])

training_size = int(len(scaled_df) * 0.8)
training, testing = scaled_df[0:training_size], scaled_df[training_size:len(scaled_df)]
print('Size of the dataset: %d' % (len(scaled_df)))
print('Training examples: %d' % (len(training)))
print('Testing examples: %d' % (len(testing)))

# circa una lettura ogni 5 minuti
# utlizzando le ultime 3 ore prediciamo la prossima ora

TRAINING_WINDOW_SIZE = 144                   # ----------- 3H
TRAINING_FUTURE_STEPS = 36                   # ----------- 3H

X_train, y_train = create_sequence(training, training, TRAINING_WINDOW_SIZE, TRAINING_FUTURE_STEPS)
X_test, y_test  = create_sequence(testing, training, TRAINING_WINDOW_SIZE, TRAINING_FUTURE_STEPS)

print("shapes ----")
print('X_train: {}'.format(X_train.shape))
print('y_train: {}'.format(y_train.shape))
print('X_test: {}'.format(X_test.shape))
print('y_test: {}'.format(y_test.shape))

# Building the model_bi
dnn_model = Sequential()
bi_lstm_model = Sequential()
lstm_model = Sequential()

# if os.path.exists("{}/dnn".format(models_path)):  
#     dnn_model = load_model("{}/dnn".format(models_path))
# else:
#     # Adding DNN layer
#     dnn_model.add(Dense(128, kernel_initializer='normal',input_shape=(X_train.shape[1], X_train.shape[-1]), activation='relu'))
#     dnn_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
#     dnn_model.add(Dense(3, kernel_initializer='normal',activation='linear'))
#     dnn_model.compile(loss='mse', optimizer='adam')
#     print(dnn_model.summary())
#     dnn_model.fit(X_train, y_train, batch_size=128, epochs=1500)
#     dnn_model.save("{}/dnn".format(models_path))

if os.path.exists("{}/bi_lstm".format(models_path)):  
    bi_lstm_model = load_model("{}/bi_lstm".format(models_path))
else:
    # Adding a Bidirectional LSTM layer
    bi_lstm_model.add(Bidirectional(LSTM(128,return_sequences=True, dropout=0.5, input_shape=(X_train.shape[1], X_train.shape[-1]))))
    bi_lstm_model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
    bi_lstm_model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
    bi_lstm_model.add(Bidirectional(LSTM(20, dropout=0.5)))
    bi_lstm_model.add(Dense(3))
    bi_lstm_model.compile(loss='mse', optimizer='adam')
    bi_lstm_model.fit(X_train, y_train, batch_size=128, epochs=1500)
    bi_lstm_model.save("{}/bi_lstm".format(models_path))
    print(bi_lstm_model.summary())

# if os.path.exists("{}/lstm".format(models_path)):  
#     lstm_model = load_model("{}/lstm".format(models_path))
# else:
#     # Adding LSTM layer
#     lstm_model.add(LSTM(128,return_sequences=True, dropout=0.5, input_shape=(X_train.shape[1], X_train.shape[-1])))
#     lstm_model.add(LSTM(128, return_sequences=True, dropout=0.5))
#     lstm_model.add(LSTM(128, return_sequences=True, dropout=0.5))
#     lstm_model.add(LSTM(20, dropout=0.5))
#     lstm_model.add(Dense(1))
#     lstm_model.compile(loss='mse', optimizer='adam')
#     lstm_model.fit(X_train, y_train, batch_size=128, epochs=1500)
#     lstm_model.save("{}/lstm".format(models_path))
#     print(lstm_model.summary())


# # -------------- TESTING

# print("---- DNN TESTING ----")
# predictions = dnn_model.predict(X_test)
# print(predictions)
# print("predictions.shape", predictions.shape)
# print("y_test.shape", y_test.shape)

# MSEs_CPUs = mean_squared_error(predictions, y_test)
# MAPE_CPUs = mean_absolute_percentage_error(predictions, y_test)

# print("MSEs_CPUs", MSEs_CPUs)
# print("MAPE_CPUs", MAPE_CPUs)
# print("---------------------------------------------------")

# print("---- BI-LSTM TESTING ----")
# predictions = bi_lstm_model.predict(X_test)
# print("predictions.shape", predictions.shape)
# print("y_test.shape", y_test.shape)

# MSEs_CPUs = mean_squared_error(predictions, y_test)
# MAPE_CPUs = mean_absolute_percentage_error(predictions, y_test)

# print("MSEs_CPUs", MSEs_CPUs)
# print("MAPE_CPUs", MAPE_CPUs)
# print("---------------------------------------------------")

# print("---- LSTM TESTING ----")
# predictions = lstm_model.predict(X_test)
# print("predictions.shape", predictions.shape)
# print("y_test.shape", y_test.shape)

# MSEs_CPUs = mean_squared_error(predictions, y_test)
# MAPE_CPUs = mean_absolute_percentage_error(predictions, y_test)

# print("MSEs_CPUs", MSEs_CPUs)
# print("MAPE_CPUs", MAPE_CPUs)
# print("---------------------------------------------------")


