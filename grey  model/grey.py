import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import rcParams

# 读取数据
data = pd.read_csv('肺癌_simple.csv')
years = data['year'].values
values = data['value'].values

# 灰色预测模型
class GreyModel:
    def __init__(self, data):
        self.data = data
        self.n = len(data)
        
    def fit(self):
        # 累加生成序列
        self.cum_data = np.cumsum(self.data)
        
        # 构造B矩阵和Y矩阵
        B = np.array([-0.5*(self.cum_data[i-1] + self.cum_data[i]) for i in range(1, self.n)]).reshape(-1, 1)
        B = np.hstack([B, np.ones((self.n-1, 1))])
        Y = self.data[1:].reshape(-1, 1)
        
        # 计算参数
        self.a, self.b = np.linalg.inv(B.T @ B) @ B.T @ Y
        
    def predict(self, steps):
        # 预测累加值
        pred_cum = [(self.data[0] - self.b/self.a) * np.exp(-self.a*k) + self.b/self.a for k in range(self.n + steps)]
        
        # 还原预测值
        pred = [pred_cum[0]]
        pred += [pred_cum[i] - pred_cum[i-1] for i in range(1, len(pred_cum))]
        
        return pred[:self.n], pred[self.n:]

# 模型训练与预测
model = GreyModel(values)
model.fit()
fitted_values, predicted_values = model.predict(2050-2022+1)

# 计算评价指标
def calculate_metrics(true, pred):
    # 确保比较的序列长度一致
    n = len(true)  # 获取真实值的长度
    mse = mean_squared_error(true[1:], pred[1:n])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true[1:], pred[1:n])
    mape = np.mean(np.abs((true[1:] - pred[1:n]) / true[1:])) * 100
    r2 = r2_score(true[1:], pred[1:n])
    return mse, rmse, mae, mape, r2

mse, rmse, mae, mape, r2 = calculate_metrics(values, fitted_values)

# 保存结果
result = pd.DataFrame({
    'year': range(2022, 2051),
    'predicted_value': predicted_values
})
result.to_csv('result_grey.csv', index=False)

# 在可视化代码前添加字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(years, values, 'bo-', label='原始数据')
plt.plot(years, fitted_values, 'r--', label='拟合曲线')
plt.plot(range(2022, 2051), predicted_values, 'g-.', label='预测值')
plt.legend()
plt.xlabel('年份')
plt.ylabel('值')
plt.title('灰色模型预测结果')
plt.grid(True)
plt.show()

# 输出评价指标
print(f'MSE: {mse:.8f}')
print(f'RMSE: {rmse:.8f}')
print(f'MAE: {mae:.8f}')
print(f'MAPE: {mape:.8f}%')
print(f'R-squared: {r2:.8f}')