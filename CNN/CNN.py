import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 读取数据
data = pd.read_csv('simple.csv')
years = data['year'].values
values = data['value'].values

# 2. 数据预处理
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step)])
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)

time_step = 5
X, y = create_dataset(values, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 3. 修改数据划分 - 全部用于训练
X_train, y_train = X, y  # 直接使用全部数据

# 4. 构建CNN模型
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(time_step, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 5. 修改训练代码 - 去掉validation_data
history = model.fit(X_train, y_train, epochs=100, verbose=1)

# 6. 预测 - 只保留训练集预测
train_predict = model.predict(X_train)

# 7. 预测未来值
future_years = np.arange(2022, 2051)
future_predictions = []
last_sequence = values[-time_step:].reshape(1, time_step, 1)

for year in future_years:
    pred = model.predict(last_sequence)[0][0]
    future_predictions.append(pred)
    last_sequence = np.append(last_sequence[:,1:,:], [[[pred]]], axis=1)

# 8. 保存结果
result_df = pd.DataFrame({'year': future_years, 'value': future_predictions})
result_df.to_csv('result_CNN.csv', index=False)

# 9. 修改可视化部分
plt.figure(figsize=(12,6))
plt.plot(years, values, label='原始数据')

# 只显示训练集预测
train_plot_range = years[time_step:len(train_predict)+time_step]
plt.plot(train_plot_range, train_predict, label='模型预测')

# 显示未来预测
plt.plot(future_years, future_predictions, label='未来预测')

plt.legend()
plt.xlabel('年份')
plt.ylabel('值')
plt.title('肺癌数据预测')
plt.savefig('prediction_plot.png')
plt.show()

# 10. 修改评估指标计算
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, mape, r2

# 修复长度不一致问题
# 原始数据长度减去time_step
y_true = values[time_step:len(train_predict)+time_step]
y_pred = train_predict

mse, rmse, mae, mape, r2 = calculate_metrics(y_true, y_pred)
print(f'MSE: {mse:.8f}')
print(f'RMSE: {rmse:.8f}')
print(f'MAE: {mae:.8f}')
print(f'MAPE: {mape:.8f}%')
print(f'R-squared: {r2:.8f}')
