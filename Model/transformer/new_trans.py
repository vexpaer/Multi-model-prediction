import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

os.makedirs('2011-2021', exist_ok=True)
files = [
    '10-14_years.csv', '15-19_years.csv', '20-24_years.csv', '25-29_years.csv',
    '30-34_years.csv', '35-39_years.csv', '40-44_years.csv', '45-49_years.csv',
    '5-9_years.csv', '50-54_years.csv', '55-59_years.csv', '60-64_years.csv',
    '65-69_years.csv', '70-74_years.csv', '75-79_years.csv', '80-84_years.csv',
    '85-89_years.csv', '90-94_years.csv', '95plus_years.csv', 'less_5_years.csv'
]

def create_dataset(data, look_back=15):
    X, y = [], []
    for i in range(len(data)-look_back):
        X.append(data[i:(i+look_back), 0])
        y.append(data[i+look_back, 0])
    return np.array(X), np.array(y)

def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(15, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

for file in files:
    df = pd.read_csv(f'rawData/{file}')
    data = df['value'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    train_data = scaled_data[:31]  
    test_data = scaled_data[31:]   
    X_train, y_train = create_dataset(train_data)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = build_model()
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
    predictions = []
    last_sequence = train_data[-15:]
    for i in range(11):  
        last_sequence = np.reshape(last_sequence, (1, 15, 1))
        pred = model.predict(last_sequence, verbose=0)
        predictions.append(pred[0][0])
        last_sequence = np.append(last_sequence[0][1:], pred)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    result_df = pd.DataFrame({
        'year': df['year'][31:].values,
        'actual': df['value'][31:].values,
        'predicted': predictions.flatten()
    })
    result_df.to_csv(f'2011-2021/result_{file}', index=False)