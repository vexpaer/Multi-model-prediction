import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
os.makedirs('2022-2050', exist_ok=True)
def create_dataset(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
def build_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(15, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
files = [f for f in os.listdir('rawData') if f.endswith('.csv')]

for file in files:
    df = pd.read_csv(f'rawData/{file}')
    data = df['value'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X_train, y_train = create_dataset(scaled_data)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = build_model()
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
    predictions = []
    last_sequence = scaled_data[-15:]
    for i in range(29):  
        last_sequence = np.reshape(last_sequence, (1, 15, 1))
        pred = model.predict(last_sequence, verbose=0)
        predictions.append(pred[0][0])
        last_sequence = np.append(last_sequence[0][1:], pred)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    result_df = pd.DataFrame({
        'year': range(2022, 2051),
        'predicted': predictions.flatten()
    })
    result_df.to_csv(f'2022-2050/result_{file}', index=False)

print("done process, result in'2022-2050'folder ")