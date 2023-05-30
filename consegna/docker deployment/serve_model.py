from flask import Flask, request, jsonify

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Bidirectional, LSTM, Input, RepeatVector, TimeDistributed
from keras.models import Model, load_model
import os

n_past = 35
n_future = 22 
n_features = 3

DATA_PATH = './data/CSVs/raw_data.csv'

def split_series(series, n_past, n_future):
  #
  # n_past ==> no of past observations
  #
  # n_future ==> no of future observations 
  #
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    # slicing the past and future parts of the window
    past, future = series[window_start:past_end, :], series[past_end:future_end, :]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)

def load_last_csv():
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['DATE'])
    df = df.sort_values(by=["timestamp"]) 
    df = df[1:]
    CPUs, MEMs, TEMPs, timestamps = [], [], [], []
    current_timestamp = None
    cpu, mem, temp = None, None, None
    for index, row in df.iterrows():
        if row["timestamp"] != current_timestamp and cpu != None and mem != None and temp != None: 
            #         print((cpu, mem, temp))
            CPUs.append(cpu)
            MEMs.append(mem)
            TEMPs.append(temp)
            timestamps.append(current_timestamp)
        
        if row["NAME"] == "CPU_USE":
            cpu = row["VAL"]
            
        if row["NAME"] == "MEM_USE":
            mem = row["VAL"]
            
        if row["NAME"] == "TEMP":
            temp = row["VAL"]
            
        current_timestamp = row["timestamp"]

    df_dict = {"CPUs": CPUs, "MEMs": MEMs, "TEMPs": TEMPs, "timestamps": timestamps}
    df_reshaped = pd.DataFrame.from_dict(df_dict)
            
    return df_reshaped

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify(
        message="ciao mondo"
    )

@app.route('/predict_on_sequence', methods=['POST'])
def predict():
    req = request.json
    input_seq_is_valid = False
    input_seq = None
    try:
        input_seq = np.array(req["sequence"])
        input_seq_is_valid = True
    except:
        print("Error while parsing input seq")
    if input_seq_is_valid == False:
        return jsonify(
            message = "(35,3) input matrix required, error while parsing",
            input_seq = req["sequence"] 
        )

    df = load_last_csv()
    df = df[["CPUs", "MEMs", "TEMPs"]]


    # parse sequence into DataFrame
    df_input_seq = pd.DataFrame(input_seq, columns = ['CPUs', 'MEMs', 'TEMPs'])

    # downscaling features
    scalers={}
    for i in df.columns:
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit_transform(df[i].values.reshape(-1,1))
        scalers['scaler_'+ i] = scaler
        df_input_seq[i] = scaler.transform(df_input_seq[i].values.reshape(-1,1))

    df_input_seq = df_input_seq.to_numpy().reshape(1,n_past,n_features)

    model_e1d1 = load_model("./saved_models/seq_to_seq/model_e1d1")
    pred_e1d1 = model_e1d1.predict(df_input_seq)

    # upscaling features
    for index,i in enumerate(df.columns):
        scaler = scalers['scaler_'+i]
        pred_e1d1[:,:,index]=scaler.inverse_transform(pred_e1d1[:,:,index])

    return jsonify(
        input_seq = req["sequence"],
        output_seq = pred_e1d1.tolist()
    )

# appends a list of string to file
# ------------------ NB --------------------- 
# there are no checks on new data strings
# ------------------------------------------- 
@app.route('/add_new_data', methods=['POST'])
def add_new_data():
    req = request.json
    new_data = req["new_data"]
    with open(DATA_PATH, "a") as myfile:
        for line in new_data:
            myfile.write('\n')
            myfile.write(line)
    return jsonify(
        message = "data appended to file",
        new_data = new_data
    )

# forces a new training
# ------------------ NB --------------------- 
# there is no authorization at the moment
# ------------------------------------------- 
@app.route('/retrain', methods=['POST'])
def retrain():
    df = load_last_csv()
    df_cleaned = df[["CPUs", "MEMs", "TEMPs"]]
    training_size = int(df_cleaned.shape[0] * .75) 
    testing_size = df_cleaned.shape[0] - training_size
    train_df,test_df = df_cleaned[:training_size], df_cleaned[training_size:] 

    print("train_df.shape", train_df.shape)
    print("test_df.shape", test_df.shape)

    # downscale the data
    train = train_df
    scalers={}
    for i in train_df.columns:
        scaler = MinMaxScaler(feature_range=(-1,1))
        s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
        s_s=np.reshape(s_s,len(s_s))
        scalers['scaler_'+ i] = scaler
        train[i]=s_s
    test = test_df
    for i in train_df.columns:
        scaler = scalers['scaler_'+i]
        s_s = scaler.transform(test[i].values.reshape(-1,1))
        s_s=np.reshape(s_s,len(s_s))
        scalers['scaler_'+i] = scaler
        test[i]=s_s

    X_train, y_train = split_series(train.values,n_past, n_future)
    X_test, y_test = split_series(test.values,n_past, n_future)

    # best model architecture
    # E1D1
    encoder_inputs = Input(shape=(n_past, n_features))
    encoder_l1 = LSTM(128, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)

    encoder_states1 = encoder_outputs1[1:]

    #
    decoder_inputs = RepeatVector(n_future)(encoder_outputs1[0])

    #
    decoder_l1 = LSTM(128, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
    decoder_outputs1 = TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)

    #
    model_e1d1 = Model(encoder_inputs,decoder_outputs1)

    #
    model_e1d1.summary()

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)

    model_e1d1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
    history_e1d1=model_e1d1.fit(X_train,y_train,epochs=500,validation_data=(X_test,y_test),batch_size=32,verbose=1,callbacks=[reduce_lr])

    model_e1d1.save("./saved_models/seq_to_seq/model_e1d1")

    return jsonify(
        message = "new training done"
    )


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)