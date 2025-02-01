import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 在导入matplotlib后添加以下代码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 读取数据
df = pd.read_csv('肺癌_simple.csv')
df.columns = ['ds', 'y']  # Prophet要求列名为ds和y
df['ds'] = pd.to_datetime(df['ds'], format='%Y')  # 将年份转换为日期格式

# 2. 创建并训练Prophet模型
model = Prophet()
model.fit(df)

# 3. 创建未来日期（2022-2050）
future = model.make_future_dataframe(periods=29, freq='Y')

# 4. 进行预测
forecast = model.predict(future)

# 5. 保存预测结果
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('result_prophet.csv', index=False)

# 6. 可视化
fig = model.plot(forecast)
plt.title('肺癌数据预测')
plt.xlabel('年份')
plt.ylabel('值')
plt.show()

# 7. 计算模型评估指标
# 获取1991年之后的数据
df_eval = df[df['ds'] >= pd.to_datetime('1992-01-01')]  # 使用pd.to_datetime转换比较值
forecast_eval = forecast[forecast['ds'].isin(df_eval['ds'])]

# 计算各项指标
mse = mean_squared_error(df_eval['y'], forecast_eval['yhat'])
rmse = mse ** 0.5
mae = mean_absolute_error(df_eval['y'], forecast_eval['yhat'])
mape = (abs((df_eval['y'] - forecast_eval['yhat']) / df_eval['y'])).mean() * 100
r2 = r2_score(df_eval['y'], forecast_eval['yhat'])

# 打印评估结果
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.4f}%')
print(f'R-squared: {r2:.4f}')