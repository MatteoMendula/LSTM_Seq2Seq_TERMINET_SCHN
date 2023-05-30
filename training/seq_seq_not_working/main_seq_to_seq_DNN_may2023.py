import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from keras.models import Sequential, save_model, load_model
from keras.layers import Bidirectional, LSTM, Dropout, Dense, Flatten
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_percentage_error
import os

def create_sequence(dataset, dim_window_input, dim_window_output):
    x_sequence, y_sequence = [], []
    print("dataset.shape", dataset.shape)
    for index in range(len(dataset) - dim_window_input - dim_window_output + 1):
        x_sequence.append(dataset[index: index + dim_window_input])
        y_sequence.append(dataset[index + dim_window_input: index + dim_window_input + dim_window_output])
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

# read dataset augmented
df_reshaped = pd.read_pickle("../data/20230319_RTU_Dataset_PPC-Lab/combined_may2023.pkl")  

# Normalizing the values
standard_scaler = preprocessing.StandardScaler()
scaled_df = standard_scaler.fit_transform(df_reshaped[['MEM_USAGE', 'CPU_USAGE', 'TEMP']])

training_size = int(len(scaled_df) * 0.8)
training, testing = scaled_df[0:training_size], scaled_df[training_size:len(scaled_df)]
print('Size of the dataset: %d' % (len(scaled_df)))
print('Training examples: %d' % (len(training)))
print('Testing examples: %d' % (len(testing)))

TRAINING_WINDOW_SIZE = 35
TRAINING_FUTURE_STEPS = 22                   # ----------- 1H

X_train, y_train = create_sequence(training, TRAINING_WINDOW_SIZE, TRAINING_FUTURE_STEPS)
X_test, y_test  = create_sequence(testing, TRAINING_WINDOW_SIZE, TRAINING_FUTURE_STEPS)

print('X_train: %s', X_train.shape)
print('y_train: %s', y_train.shape)
print('X_test: %s', X_test.shape)
print('y_test: %s', y_test.shape)

# Building the model_bi
model_dnn = Sequential()
model_lstm = Sequential()
model_bi_lstm = Sequential()

# if os.path.exists("./saved_models/seq_to_seq/model_dnn"):  
#     model_dnn = load_model("./saved_models/seq_to_seq/model_dnn")
# else:
#     # Adding DNN layer
#     model_dnn.add(Dense(128, kernel_initializer='normal',input_shape=(X_train.shape[1], X_train.shape[-1]), activation='relu'))
#     model_dnn.add(Dense(128, kernel_initializer='normal',activation='relu'))
#     model_dnn.add(Dense(3, kernel_initializer='normal',activation='linear'))
#     model_dnn.compile(loss='mean_absolute_error', optimizer='adam')
#     print(model_dnn.summary())
#     model_dnn.fit(X_train, y_train, batch_size=128, epochs=500)
#     model_dnn.save("./saved_models/seq_to_seq/model_dnn")

if os.path.exists("./saved_models/seq_to_seq/model_lstm"):  
    model_lstm = load_model("./saved_models/seq_to_seq/model_lstm")
else:
    # Adding LSTM layer
    model_lstm.add(LSTM(128,return_sequences=True, dropout=0.5, input_shape=(X_train.shape[1], X_train.shape[-1])))
    # model_lstm.add(LSTM(128, return_sequences=True, dropout=0.5))
    # model_lstm.add(LSTM(128, return_sequences=True, dropout=0.5))
    # model_lstm.add(LSTM(20, dropout=0.5))
    model_lstm.add(Dense(3))
    model_lstm.compile(loss='mse', optimizer='adam')
    model_lstm.fit(X_train, y_train, batch_size=128, epochs=500)
    model_lstm.save("./saved_models/seq_to_seq/model_lstm")
    print(model_lstm.summary())

if os.path.exists("./saved_models/seq_to_seq/model_bi_lstm"):  
    model_bi_lstm = load_model("./saved_models/seq_to_seq/model_bi_lstm")
else:
    # Adding a Bidirectional LSTM layer
    model_bi_lstm.add(Bidirectional(LSTM(128,return_sequences=True, dropout=0.5, input_shape=(X_train.shape[1], X_train.shape[-1]))))
    model_bi_lstm.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
    model_bi_lstm.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
    model_bi_lstm.add(Bidirectional(LSTM(20, dropout=0.5)))
    model_bi_lstm.add(Dense(3))
    model_bi_lstm.compile(loss='mse', optimizer='adam')
    model_bi_lstm.fit(X_train, y_train, batch_size=128, epochs=500)
    model_bi_lstm.save("./saved_models/seq_to_seq/model_bi_lstm")
    print(model_bi_lstm.summary())


# -------------- TESTING

# predictions_dnn = model_dnn.predict(X_test)
predictions_lstm = model_lstm.predict(X_test)
predictions_bi_lstm = model_bi_lstm.predict(X_test)

# predictions_dnn = predictions_dnn.reshape(predictions_dnn.shape[0], predictions_dnn.shape[2])

# print("predictions_dnn.shape", predictions_dnn.shape)
print("predictions_lstm.shape", predictions_lstm.shape)
print("predictions_bi_lstm.shape", predictions_bi_lstm.shape)

y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])
y_test = standard_scaler.inverse_transform(y_test)

# predictions_dnn = standard_scaler.inverse_transform(predictions_dnn)
predictions_lstm = standard_scaler.inverse_transform(predictions_lstm)
predictions_bi_lstm = standard_scaler.inverse_transform(predictions_bi_lstm)

CPUs_prediction_lstm = predictions_lstm[:, 0] 
MEMs_prediction_lstm = predictions_lstm[:, 1] 
TEMPs_prediction_lstm = predictions_lstm[:, 2] 

CPUs_prediction_bi_lstm = predictions_bi_lstm[:, 0] 
MEMs_prediction_bi_lstm = predictions_bi_lstm[:, 1] 
TEMPs_prediction_bi_lstm = predictions_bi_lstm[:, 2] 

# CPUs_prediction_dnn = predictions_dnn[:, 0] 
# MEMs_prediction_dnn = predictions_dnn[:, 1] 
# TEMPs_prediction_dnn = predictions_dnn[:, 2] 

CPUs_y_test = y_test[:, 0] 
MEMs_y_test = y_test[:, 1] 
TEMPs_y_test = y_test[:, 2] 

# ------------------ CPU error evaluation

# MSEs_CPUs_dnn = mean_squared_error(CPUs_prediction_dnn, CPUs_y_test)
# MAPE_CPUs_dnn = mean_absolute_percentage_error(CPUs_prediction_dnn, CPUs_y_test)

MSEs_CPUs_lstm = mean_squared_error(CPUs_prediction_lstm, CPUs_y_test)
MAPE_CPUs_lstm = mean_absolute_percentage_error(CPUs_prediction_lstm, CPUs_y_test)

MSEs_CPUs_bi_lstm = mean_squared_error(CPUs_prediction_bi_lstm, CPUs_y_test)
MAPE_CPUs_bi_lstm = mean_absolute_percentage_error(CPUs_prediction_bi_lstm, CPUs_y_test)

print("------- CPU ----------")
# print("MSEs_CPUs_dnn", MSEs_CPUs_dnn)
# print("MAPE_CPUs_dnn", MAPE_CPUs_dnn)
# print("----------------------")
print("MSEs_CPUs_lstm", MSEs_CPUs_lstm)
print("MAPE_CPUs_lstm", MAPE_CPUs_lstm)
print("----------------------")
print("MSEs_CPUs_bi_lstm", MSEs_CPUs_bi_lstm)
print("MAPE_CPUs_bi_lstm", MAPE_CPUs_bi_lstm)
print("----------------------")


# ------------------ MEMs error evaluation

MSEs_MEMs_lstm = mean_squared_error(MEMs_prediction_lstm, MEMs_y_test)
MAPE_MEMs_lstm = mean_absolute_percentage_error(MEMs_prediction_lstm, MEMs_y_test)

MSEs_TEMPs_bi_lstm = mean_squared_error(MEMs_prediction_bi_lstm, TEMPs_y_test)
MAPE_TEMPs_bi_lstm = mean_absolute_percentage_error(MEMs_prediction_bi_lstm, TEMPs_y_test)

print("------- MEMs ----------")
# print("MSEs_CPUs_bi", MSEs_CPUs_bi)
# print("MAPE_CPUs_bi", MAPE_CPUs_bi)
# print("-----------------")
print("MSEs_MEMs_lstm", MSEs_MEMs_lstm)
print("MAPE_MEMs_lstm", MAPE_MEMs_lstm)
print("-----------------")
print("MSEs_TEMPs_bi_lstm", MSEs_TEMPs_bi_lstm)
print("MAPE_TEMPs_bi_lstm", MAPE_TEMPs_bi_lstm)
print("-----------------")


# ------------------ TEMPs error evaluation

MSEs_TEMPs_lstm = mean_squared_error(TEMPs_prediction_lstm, TEMPs_y_test)
MAPE_TEMPs_lstm = mean_absolute_percentage_error(TEMPs_prediction_lstm, TEMPs_y_test)

MSEs_MEMs_bi_lstm = mean_squared_error(MEMs_prediction_bi_lstm, MEMs_y_test)
MAPE_MEMs_bi_lstm = mean_absolute_percentage_error(MEMs_prediction_bi_lstm, MEMs_y_test)

print("------- TEMPs ----------")
# print("MSEs_CPUs_bi", MSEs_CPUs_bi)
# print("MAPE_CPUs_bi", MAPE_CPUs_bi)
# print("-----------------")
print("MSEs_TEMPs_lstm", MSEs_TEMPs_lstm)
print("MAPE_TEMPs_lstm", MAPE_TEMPs_lstm)
print("-----------------")
print("MSEs_MEMs_bi_lstm", MSEs_MEMs_bi_lstm)
print("MAPE_MEMs_bi_lstm", MAPE_MEMs_bi_lstm)
print("-----------------")
# print(find_max_error(predictions, y_test, np.mean(MSEs_CPUs), np.std(MSEs_CPUs)))

df_predictions_lstm = pd.DataFrame(predictions_lstm, columns = ['CPUs', 'MEMs', 'TEMPs'])
df_predictions_bi_lstm = pd.DataFrame(predictions_bi_lstm, columns = ['CPUs', 'MEMs', 'TEMPs'])

df_predictions_lstm.to_pickle("./predictions/seq_to_seq/df_predictions_lstm.pkl")
df_predictions_bi_lstm.to_pickle("./predictions/seq_to_seq/predictions.pkl")

# for index in outliers.index: 
#     outliers[index] = predictions[index]
# print("outliers", outliers)

# Showing the predicted vs. actual values
# fig, axs = plt.subplots()
# fig.set_figheight(4)
# fig.set_figwidth(15)

# axs.plot(predictions,color='red', label='Predicted')
# axs.plot(y_test,color='blue', label='Actual')
# axs.scatter(outliers.index,outliers, color='green', linewidth=5.0, label='Anomalies')
# plt.xlabel('Timestamp')
# plt.ylabel('Scaled number of passengers')
# plt.legend(loc='upper left')
# plt.show()