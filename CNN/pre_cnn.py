import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def process_file(file_path):
    df = pd.read_csv(file_path)
    values = df['value'].values
    def create_dataset(data, look_back=10):
        X, y = [], []
        for i in range(len(data)-look_back):
            X.append(data[i:i+look_back])
            y.append(data[i+look_back])
        return np.array(X), np.array(y)
    X_train, y_train = create_dataset(values)  
    last_window = values[-10:]
    future_predictions = []
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    model = create_cnn_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=100, verbose=0)
    
    current_window = last_window.copy()
    for year in range(2022, 2051):
        input_data = current_window.reshape((1, 10, 1))
        prediction = model.predict(input_data, verbose=0)[0][0]
        future_predictions.append(prediction)
        current_window = np.append(current_window[1:], prediction)
    
    result_df = pd.DataFrame({
        'year': range(2022, 2051),
        'predicted': future_predictions
    })
    return result_df

if __name__ == '__main__':
    os.makedirs('2022-2050', exist_ok=True)
    data_dir = 'rawData'
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    for file in files:
        file_path = os.path.join(data_dir, file)
        result = process_file(file_path)
        output_path = os.path.join('2022-2050', f'result_{file}')
        result.to_csv(output_path, index=False)
        print(f'{file} done process, result in {output_path}')