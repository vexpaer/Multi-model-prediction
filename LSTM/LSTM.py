import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 读取数据
file_name = "肺癌.csv"
data = pd.read_csv(file_name)

# 检查数据格式
data.columns = ['year', 'value', 'Lower bound', 'Upper bound']
data = data[['year', 'value']]

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
data['value_scaled'] = scaler.fit_transform(data[['value']])

# 创建时间序列数据
def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(x), np.array(y)

sequence_length = 5
values = data['value_scaled'].values
x, y = create_sequences(values, sequence_length)

# 拓展维度用于LSTM
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# 构建LSTM模型
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(x, y, batch_size=1, epochs=50, verbose=1)

# 预测未来数据
future_years = list(range(2022, 2051))
future_predictions = []

last_sequence = values[-sequence_length:].tolist()
for _ in future_years:
    input_sequence = np.array(last_sequence[-sequence_length:]).reshape(1, sequence_length, 1)
    next_value = model.predict(input_sequence)[0, 0]
    future_predictions.append(next_value)
    last_sequence.append(next_value)

# 反转缩放
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# 保存结果
result_df = pd.DataFrame({
    'year': future_years,
    'predicted_value': future_predictions
})
result_df.to_csv('result.csv', index=False)

print("预测完成，结果已保存到 result1.csv 文件中！")
