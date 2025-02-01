import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# 1. 数据加载和预处理
data = pd.read_csv('肺癌_simple.csv')
values = data['value'].values.reshape(-1, 1)

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# 2. 创建训练数据集
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 3
X, Y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 3. 构建RNN模型
model = Sequential()
model.add(SimpleRNN(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(SimpleRNN(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, Y, epochs=100, batch_size=1, verbose=2)

# 4. 预测和评估
train_predict = model.predict(X)
train_predict = scaler.inverse_transform(train_predict)
Y_actual = scaler.inverse_transform(Y.reshape(-1, 1))

# 计算指标
mse = mean_squared_error(Y_actual[1:], train_predict[1:])
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_actual[1:], train_predict[1:])
mape = np.mean(np.abs((Y_actual[1:] - train_predict[1:]) / Y_actual[1:])) * 100
r2 = r2_score(Y_actual[1:], train_predict[1:])

print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.4f}%')
print(f'R-squared: {r2:.4f}')

# 5. 预测未来值 (2022-2050)
future_years = 29
last_sequence = scaled_data[-time_step:]
predictions = []

for i in range(future_years):
    pred = model.predict(last_sequence.reshape(1, time_step, 1))
    predictions.append(pred[0, 0])
    last_sequence = np.append(last_sequence[1:], pred)

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 6. 保存结果
result_df = pd.DataFrame({
    'year': range(2022, 2051),
    'value': predictions.flatten()
})
result_df.to_csv('result_RNN.csv', index=False)

# 7. 可视化
plt.figure(figsize=(12, 6))
plt.plot(data['year'], data['value'], label='原始数据')
plt.plot(data['year'][time_step+1:], train_predict, label='拟合曲线')
plt.plot(range(2022, 2051), predictions, label='预测值')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('肺癌数据预测')
plt.show()
