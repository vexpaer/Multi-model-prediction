import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

if not os.path.exists('2022-2050'):
    os.makedirs('2022-2050')

files = [
    '10-14_years.csv', '15-19_years.csv', '20-24_years.csv',
    '25-29_years.csv', '30-34_years.csv', '35-39_years.csv',
    '40-44_years.csv', '45-49_years.csv', '5-9_years.csv',
    '50-54_years.csv', '55-59_years.csv', '60-64_years.csv',
    '65-69_years.csv', '70-74_years.csv', '75-79_years.csv',
    '80-84_years.csv', '85-89_years.csv', '90-94_years.csv',
    '95plus_years.csv', 'less_5_years.csv'
]

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

for file in files:
    df = pd.read_csv(f'rawData/{file}')
    data = df['value'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    train_data = scaled_data
    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)
    test_input = train_data[-time_step:]
    predictions = []
    for i in range(29):  
        test_input = test_input.reshape((1, time_step, 1))
        pred = model.predict(test_input, verbose=0)
        predictions.append(pred[0][0])
        test_input = np.append(test_input[0][1:], pred)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    result_df = pd.DataFrame({
        'year': range(2022, 2051),
        'value': predictions.flatten()
    })
    result_df.to_csv(f'2022-2050/result_{file}', index=False)