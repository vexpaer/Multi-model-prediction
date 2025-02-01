import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. 读取数据
data = pd.read_csv('肺癌_simple.csv')
data['year'] = pd.to_datetime(data['year'], format='%Y')

# 2. 准备VAR模型
model = VAR(data[['value']])

# 3. 拟合模型
results = model.fit(maxlags=2)

# 4. 预测2022-2050年数据
forecast_years = pd.date_range(start='2022', end='2050', freq='Y')
forecast = results.forecast(results.y, steps=len(forecast_years))

# 5. 保存预测结果
forecast_df = pd.DataFrame({
    'year': forecast_years.year,
    'value': forecast[:, 0]
})
forecast_df.to_csv('result_VAR.csv', index=False)

# 6. 可视化
plt.figure(figsize=(12, 6))
plt.plot(data['year'], data['value'], label='原始数据')
plt.plot(forecast_years, forecast[:, 0], label='预测数据', linestyle='--')
plt.title('肺癌数据预测')
plt.xlabel('年份')
plt.ylabel('值')
plt.legend()
plt.grid()
plt.show()

# 7. 模型评估
train_data = data[data['year'].dt.year >= 1991]
predictions = results.fittedvalues

mse = mean_squared_error(train_data['value'], predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(train_data['value'], predictions)
mape = np.mean(np.abs((train_data['value'] - predictions) / train_data['value'])) * 100
r2 = r2_score(train_data['value'], predictions)

print(f'MSE: {mse:.8f}')
print(f'RMSE: {rmse:.8f}')
print(f'MAE: {mae:.8f}')
print(f'MAPE: {mape:.8f}%')
print(f'R-squared: {r2:.8f}')
