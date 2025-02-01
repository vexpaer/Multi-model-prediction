import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 创建预测结果文件夹
os.makedirs('预测结果', exist_ok=True)

# 创建数据集
def create_dataset(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# 构建LSTM模型
def build_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(15, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# 获取数据文件列表
files = [f for f in os.listdir('原数据') if f.endswith('.csv')]

for file in files:
    # 读取数据
    df = pd.read_csv(f'原数据/{file}')
    data = df['value'].values.reshape(-1, 1)
    
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 使用全部数据训练
    X_train, y_train = create_dataset(scaled_data)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # 创建模型并训练
    model = build_model()
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
    
    # 递归预测2022-2050年
    predictions = []
    last_sequence = scaled_data[-15:]
    for i in range(29):  # 预测29年（2022-2050）
        last_sequence = np.reshape(last_sequence, (1, 15, 1))
        pred = model.predict(last_sequence, verbose=0)
        predictions.append(pred[0][0])
        last_sequence = np.append(last_sequence[0][1:], pred)
    
    # 反归一化
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # 保存结果
    result_df = pd.DataFrame({
        'year': range(2022, 2051),
        'predicted': predictions.flatten()
    })
    result_df.to_csv(f'预测结果/result_{file}', index=False)

print("预测完成，结果已保存至'预测结果'文件夹")