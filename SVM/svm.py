import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 读取数据
data = pd.read_csv('肺癌_simple.csv')
years = data['year'].values.reshape(-1, 1)
values = data['value'].values

# 划分训练集和测试集
train_years = years[:-1]  # 1991-2021
train_values = values[:-1]
test_year = years[-1:]    # 2022
test_value = values[-1:]

# 创建并训练SVM模型
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(train_years, train_values)

# 预测2022-2050年数据
future_years = np.arange(2022, 2051).reshape(-1, 1)
predicted_values = svr.predict(future_years)

# 保存预测结果
result_df = pd.DataFrame({
    'year': future_years.flatten(),
    'predicted_value': predicted_values
})
result_df.to_csv('result_svm.csv', index=False)

# 计算评估指标
y_pred = svr.predict(train_years[1:])  # 从1992年开始
y_true = train_values[1:]

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)

print(f'MSE: {mse:.8f}')
print(f'RMSE: {rmse:.8f}')
print(f'MAE: {mae:.8f}')
print(f'MAPE: {mape:.8f}%')
print(f'R-squared: {r2:.8f}')

# 可视化
plt.figure(figsize=(12, 6))

# 设置中文字体
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
# rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # Linux系统

# 获取拟合曲线
fitted_values = svr.predict(train_years)

# 绘制图形
plt.scatter(years, values, color='blue', label='原始数据')
plt.plot(train_years, fitted_values, color='green', label='拟合曲线')  # 新增拟合曲线
plt.plot(future_years, predicted_values, color='red', label='预测曲线')
plt.xlabel('年份')
plt.ylabel('值')
plt.title('肺癌数据拟合与预测')
plt.legend()
plt.grid(True)
plt.show() 