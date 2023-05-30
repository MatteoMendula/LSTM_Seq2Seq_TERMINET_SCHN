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

def create_sequence(dataset, target, window, future):
    x_sequence, y_sequence = [], []
    for index in range(len(dataset) - window - future):
        x_sequence.append(dataset[index: index + window])
        y_sequence.append(target[index + window + future: index + window + future + 1])
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
df_reshaped = pd.read_pickle("../data/PKLs/df_reshaped.pkl")  

# Normalizing the values
standard_scaler = preprocessing.StandardScaler()
print(df_reshaped.head())
scaled_df = standard_scaler.fit_transform(df_reshaped[['CPUs', 'MEM_USEs', 'TEMPs']])

training_size = int(len(scaled_df) * 0.8)
training, testing = scaled_df[0:training_size], scaled_df[training_size:len(scaled_df)]
print('Size of the dataset: %d' % (len(scaled_df)))
print('Training examples: %d' % (len(training)))
print('Testing examples: %d' % (len(testing)))

TRAINING_WINDOW_SIZE = 20
TRAINING_FUTURE_STEPS = 1                   # ----------- 1H

X_train, y_train = create_sequence(training, df_reshaped['CPUs'], TRAINING_WINDOW_SIZE, TRAINING_FUTURE_STEPS)
X_test, y_test  = create_sequence(testing, df_reshaped['CPUs'], TRAINING_WINDOW_SIZE, TRAINING_FUTURE_STEPS)

print('X_train: {}'.format(X_train.shape))
print('y_train: {}'.format(y_train.shape))
print('X_test: {}'.format(X_test.shape))
print('y_test: {}'.format(y_test.shape))

# Building the model_bi
model_bi = Sequential()
model = Sequential()
if os.path.exists("./saved_models/model_bi"):  
    model_bi = load_model("./saved_models/model_bi")
else:
    # Adding a Bidirectional LSTM layer
    model_bi.add(Bidirectional(LSTM(128,return_sequences=True, dropout=0.5, input_shape=(X_train.shape[1], X_train.shape[-1]))))
    model_bi.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
    model_bi.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
    model_bi.add(Bidirectional(LSTM(20, dropout=0.5)))
    model_bi.add(Dense(1))
    model_bi.compile(loss='mse', optimizer='adam')
    # model_bi.fit(X_train, y_train, batch_size=128, epochs=100)
    # model_bi.save("./saved_models/model_bi")
    # print(model_bi.summary())

if os.path.exists("./saved_models/model"):  
    model = load_model("./saved_models/model")
else:
    # Adding LSTM layer
    model.add(LSTM(128,return_sequences=True, dropout=0.5, input_shape=(X_train.shape[1], X_train.shape[-1])))
    model.add(LSTM(128, return_sequences=True, dropout=0.5))
    model.add(LSTM(128, return_sequences=True, dropout=0.5))
    model.add(LSTM(20, dropout=0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, batch_size=128, epochs=1500)
    model.save("./saved_models/model")
    print(model.summary())


# -------------- TESTING

# predictions_bi = model_bi.predict(X_test)
predictions = model.predict(X_test)
print("predictions.shape", predictions.shape)
print("y_test.shape", y_test.shape)

# CPUs_y_test = y_test[:, 0] 
# MEMs_y_test = y_test[:, 1] 
# TEMPs_y_test = y_test[:, 2] 

# MSEs_CPUs = evaluate_predictions(CPUs_prediction, CPUs_y_test)
# print("np.mean(MSEs)", np.mean(MSEs_CPUs))
# print("np.std(MSEs)", np.std(MSEs_CPUs))

# MSEs_CPUs_bi = mean_squared_error(predictions, y_test)
# MAPE_CPUs_bi = mean_absolute_percentage_error(predictions, y_test)

# MSEs_MEMs_bi = mean_squared_error(MEMs_prediction_bi, MEMs_y_test)
# MAPE_MEMs_bi = mean_absolute_percentage_error(MEMs_prediction_bi, MEMs_y_test)

# MSEs_TEMPs_bi = mean_squared_error(TEMPs_prediction_bi, TEMPs_y_test)
# MAPE_TEMPs_bi = mean_absolute_percentage_error(TEMPs_prediction_bi, TEMPs_y_test)

# print("-----------------")
# print("MSEs_CPUs_bi", MSEs_CPUs_bi)
# print("MAPE_CPUs_bi", MAPE_CPUs_bi)
# print("-----------------")
# print("MSEs_MEMs_bi", MSEs_MEMs_bi)
# print("MAPE_MEMs_bi", MAPE_MEMs_bi)
# print("-----------------")
# print("MSEs_TEMPs_bi", MSEs_TEMPs_bi)
# print("MAPE_TEMPs_bi", MAPE_TEMPs_bi)
# print("-----------------")

MSEs_CPUs = mean_squared_error(predictions, y_test)
MAPE_CPUs = mean_absolute_percentage_error(predictions, y_test)

# MSEs_MEMs = mean_squared_error(MEMs_prediction, MEMs_y_test)
# MAPE_MEMs = mean_absolute_percentage_error(MEMs_prediction, MEMs_y_test)

# MSEs_TEMPs = mean_squared_error(TEMPs_prediction, TEMPs_y_test)
# MAPE_TEMPs = mean_absolute_percentage_error(TEMPs_prediction, TEMPs_y_test)

print("-----------------")
print("MSEs_CPUs", MSEs_CPUs)
print("MAPE_CPUs", MAPE_CPUs)
print("-----------------")
# print("MSEs_MEMs", MSEs_MEMs)
# print("MAPE_MEMs", MAPE_MEMs)
# print("-----------------")
# print("MSEs_TEMPs", MSEs_TEMPs)
# print("MAPE_TEMPs", MAPE_TEMPs)
# print("-----------------")

# print(find_max_error(predictions, y_test, np.mean(MSEs_CPUs), np.std(MSEs_CPUs)))

# df_predictions_bi = pd.DataFrame(predictions_bi, columns = ['CPUs'])
df_predictions = pd.DataFrame(predictions, columns = ['CPUs'])

# df_predictions_bi.to_pickle("./predictions/cpu/predictions_bi.pkl")
df_predictions.to_pickle("./predictions/cpu/predictions.pkl")

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