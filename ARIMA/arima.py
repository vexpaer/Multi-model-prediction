import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. 读取数据
data = pd.read_csv('simple.csv')
data['year'] = pd.to_datetime(data['year'], format='%Y')

# 2. 划分训练集和测试集
train = data[data['year'].dt.year < 2022]
test = data[data['year'].dt.year >= 2022]

# 3. 训练ARIMA模型
model = ARIMA(train['value'], order=(5,1,0))  # 可根据数据调整p,d,q参数
model_fit = model.fit()

# 4. 预测2022-2050年数据
forecast_steps = 2050 - 2022 + 1  # 29 steps
forecast = model_fit.forecast(steps=forecast_steps)

# 5. 保存预测结果
result = pd.DataFrame({
    'year': pd.date_range(start='2022-01-01', periods=forecast_steps, freq='YE'),  # 明确指定起始日期和periods
    'value': forecast
})
result.to_csv('result_arima.csv', index=False)

# 6. 可视化
plt.figure(figsize=(12, 6))
plt.plot(data['year'], data['value'], label='原始数据')
plt.plot(result['year'], result['value'], label='预测数据', color='red')
plt.legend()
plt.title('肺癌数据ARIMA预测')
plt.xlabel('年份')
plt.ylabel('值')
plt.grid()
plt.savefig('arima_prediction.png')
plt.show()

# 7. 计算模型评估指标（1991年不计入计算）
train = train[train['year'].dt.year > 1991]
predictions = model_fit.predict(start=1, end=len(train))

mse = mean_squared_error(train['value'], predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(train['value'], predictions)
mape = np.mean(np.abs((train['value'] - predictions) / train['value'])) * 100
r2 = r2_score(train['value'], predictions)

print(f'MSE: {mse:.8f}')
print(f'RMSE: {rmse:.8f}')
print(f'MAE: {mae:.8f}')
print(f'MAPE: {mape:.8f}%')
print(f'R-squared: {r2:.8f}')