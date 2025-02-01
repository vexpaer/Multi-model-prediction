import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt

# 创建RNN模型
def create_rnn_model():
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# 处理单个文件
def process_file(file_name):
    # 读取数据
    df = pd.read_csv(os.path.join(input_folder, file_name))
    values = df['value'].values.reshape(-1, 1)
    
    # 数据归一化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(values)
    
    # 准备训练数据（1980-2021年）
    train_data = scaled_data[:42]  # 1980-2021年数据
    X_train = []
    y_train = []
    for i in range(41):  # 使用42年数据
        X_train.append(train_data[i:i+1])
        y_train.append(train_data[i+1])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # 训练模型
    model = create_rnn_model()
    model.fit(X_train, y_train, epochs=100, verbose=0)
    
    # 预测2022-2050年数据
    predictions = []
    last_value = train_data[-1]
    for _ in range(29):  # 预测29年（2022-2050）
        next_value = model.predict(last_value.reshape(1, 1, 1))
        predictions.append(next_value[0][0])
        last_value = next_value
    
    # 反归一化
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # 保存预测结果
    result_df = pd.DataFrame({
        'year': range(2022, 2051),
        'predicted': predictions.flatten()
    })
    result_df.to_csv(os.path.join(output_folder, f'result_{file_name}'), index=False)

# 主程序
if __name__ == '__main__':
    # 文件夹设置
    input_folder = '原数据'
    output_folder = '预测结果'
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 文件列表
    file_names = [
        '10-14_years.csv', '15-19_years.csv', '20-24_years.csv',
        '25-29_years.csv', '30-34_years.csv', '35-39_years.csv',
        '40-44_years.csv', '45-49_years.csv', '5-9_years.csv',
        '50-54_years.csv', '55-59_years.csv', '60-64_years.csv',
        '65-69_years.csv', '70-74_years.csv', '75-79_years.csv',
        '80-84_years.csv', '85-89_years.csv', '90-94_years.csv',
        '95plus_years.csv', 'less_5_years.csv'
    ]
    
    # 处理所有文件
    for file_name in file_names:
        print(f'Processing {file_name}...')
        process_file(file_name)
    
    print('All files processed successfully!')