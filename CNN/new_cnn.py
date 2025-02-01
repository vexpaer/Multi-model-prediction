import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def process_file(file_path):
    df = pd.read_csv(file_path)
    values = df['value'].values
    train_data = values[:31] 
    test_data = values[31:]  
    
    def create_dataset(data, look_back=10): 
        X, y = [], []
        for i in range(len(data)-look_back):
            X.append(data[i:i+look_back])
            y.append(data[i+look_back])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_dataset(values[:31]) 
    X_test, y_test = create_dataset(values[21:])   
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = create_cnn_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=100, verbose=0)
    
    predictions = model.predict(X_test)
    
    result_df = pd.DataFrame({
        'year': df['year'][-len(predictions):],
        'actual': y_test,
        'predicted': predictions.flatten()
    })
    return result_df

if __name__ == '__main__':
    os.makedirs('2011-2021', exist_ok=True)
    
    files = [
        '10-14_years.csv', '15-19_years.csv', '20-24_years.csv',
        '25-29_years.csv', '30-34_years.csv', '35-39_years.csv',
        '40-44_years.csv', '45-49_years.csv', '5-9_years.csv',
        '50-54_years.csv', '55-59_years.csv', '60-64_years.csv',
        '65-69_years.csv', '70-74_years.csv', '75-79_years.csv',
        '80-84_years.csv', '85-89_years.csv', '90-94_years.csv',
        '95plus_years.csv', 'less_5_years.csv'
    ]
    
    for file in files:
        file_path = os.path.join('rawData', file)
        result = process_file(file_path)
        output_path = os.path.join('2011-2021', f'result_{file}')
        result.to_csv(output_path, index=False)
        print(f'{file} done process,result in {output_path}')