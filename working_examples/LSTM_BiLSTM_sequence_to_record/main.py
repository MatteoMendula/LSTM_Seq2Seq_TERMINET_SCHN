import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dropout, Dense

df = pd.read_csv('data/nyc_taxi.csv', engine='python')

# Convert the timestamp string to datetime datatype (year, month, day, hour)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['Year'] = df.timestamp.dt.year
df['Month'] = df.timestamp.dt.month
df['Day'] = df.timestamp.dt.day
df['Hour'] = df.timestamp.dt.hour

df['day_time'] = ((df['Hour'] >= 6) & (df['Hour'] <= 22)).astype(int)   
df['Weekday'] = df.timestamp.dt.weekday
df['avg_hour_day'] = df.Weekday.astype(str) + ' ' + df.Hour.astype(str)
df.avg_hour_day = df.avg_hour_day.replace(df[:7344].groupby(df.Weekday.astype(str) + ' ' + df.Hour.astype(str))['value'].mean().to_dict())

standard_scaler = preprocessing.StandardScaler()
scaled_data = standard_scaler.fit_transform(df[['Hour', 'day_time', 'Weekday', 'avg_hour_day', 'value']])
scaled_df = df.copy()
scaled_df['Hour'] = scaled_data[:,0]
scaled_df['day_time'] = scaled_data[:,1]
scaled_df['Weekday'] = scaled_data[:,2]
scaled_df['avg_hour_day'] = scaled_data[:,3]
scaled_df['value'] = scaled_data[:,4]
scaled_df.head(3)

# Specifying how many values to predict
time_step = 1 

training_size = int(len(scaled_df) * 0.9)
training, testing = scaled_df[0:training_size], scaled_df[training_size:len(df)]
print('Size of the dataset: %d' % (len(scaled_df)))
print('Training examples: %d' % (len(training)))
print('Testing examples: %d' % (len(testing)))

# training features: Value, Hour, day_time
X_train = training[['value', 'Hour', 'day_time']].to_numpy()
y_train = scaled_df[time_step:testing.index[0]]['value'].to_numpy()

# testing data
X_test = testing[0:-time_step][['value', 'Hour', 'day_time']].to_numpy()
y_test = scaled_df[testing.index[0] + time_step:]['value'].to_numpy()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(training['timestamp'], X_train[:,0])
ax.plot(testing['timestamp'][0:-1], X_test[:,0])
ax.set_title('NYC Taxi Demand')
plt.xlabel('timestamp')
plt.ylabel('Scaled Number passengers')
plt.show()

# create sequences of (48-two readings per hour) data points for each training example
def create_sequence(dataset, length):
    data_sequences = []
    for index in range(len(dataset) - length):
        data_sequences.append(dataset[index: index + length])
    return np.asarray(data_sequences)
    
def evaluate_predictions(predictions, y_test, outliers):
    ratio = []
    differences = []
    for pred in range(len(y_test)):
        ratio.append((y_test[pred]/predictions[pred])-1)
        differences.append(abs(y_test[pred]- predictions[pred]))
    
    
    n_outliers = int(len(differences) * outliers)
    outliers = pd.Series(differences).astype(float).nlargest(n_outliers)
    
    return ratio, differences, outliers

X_train = create_sequence(X_train, 48)
X_test  = create_sequence(X_test, 48)
y_train = y_train[-X_train.shape[0]:]
y_test  = y_test[-X_test.shape[0]:]

# Building the model
model = Sequential()
# Adding a Bidirectional LSTM layer
model.add(Bidirectional(LSTM(64,return_sequences=True, dropout=0.5, input_shape=(X_train.shape[1], X_train.shape[-1]))))
model.add(Bidirectional(LSTM(20, dropout=0.5)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

# Training the model
model.fit(X_train, y_train, batch_size=128, epochs=50)

model.summary()

# create the list of difference between prediction and test data
predictions = model.predict(X_test)
len(predictions)

ratio, differences, outliers = evaluate_predictions(predictions, y_test, 0.01)

for index in outliers.index: 
    outliers[index] = predictions[index]
outliers

# Showing the predicted vs. actual values
fig, axs = plt.subplots()
fig.set_figheight(4)
fig.set_figwidth(15)

axs.plot(predictions,color='red', label='Predicted')
axs.plot(y_test,color='blue', label='Actual')
axs.scatter(outliers.index,outliers, color='green', linewidth=5.0, label='Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('Scaled number of passengers')
plt.legend(loc='upper left')
plt.show()