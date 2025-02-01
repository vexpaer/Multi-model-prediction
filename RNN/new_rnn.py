import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# 定义文件路径
input_folder = '原数据'
output_folder = '年龄结果'
file_names = [
    '10-14_years.csv', '15-19_years.csv', '20-24_years.csv', '25-29_years.csv',
    '30-34_years.csv', '35-39_years.csv', '40-44_years.csv', '45-49_years.csv',
    '5-9_years.csv', '50-54_years.csv', '55-59_years.csv', '60-64_years.csv',
    '65-69_years.csv', '70-74_years.csv', '75-79_years.csv', '80-84_years.csv',
    '85-89_years.csv', '90-94_years.csv', '95plus_years.csv', 'less_5_years.csv'
]

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义RNN模型
def create_rnn_model():
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(30, 1)))
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
    
    # 准备训练数据
    train_data = scaled_data[:31]  # 1980-2010年数据
    X_train = []
    y_train = []
    for i in range(30):
        X_train.append(train_data[i:i+1])
        y_train.append(train_data[i+1])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # 训练模型
    model = create_rnn_model()
    model.fit(X_train, y_train, epochs=100, verbose=0)
    
    # 预测2011-2021年数据
    predictions = []
    last_value = train_data[-1]
    for _ in range(11):
        next_value = model.predict(last_value.reshape(1, 1, 1))
        predictions.append(next_value[0][0])
        last_value = next_value
    
    # 反归一化
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # 保存结果
    result_df = pd.DataFrame({
        'year': range(2011, 2022),
        'actual': df['value'].values[31:42],
        'predicted': predictions.flatten()
    })
    result_df.to_csv(os.path.join(output_folder, f'result_{file_name}'), index=False)

# 处理所有文件
for file_name in file_names:
    process_file(file_name)
    print(f'{file_name} 处理完成')