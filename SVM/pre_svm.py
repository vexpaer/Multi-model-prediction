import os
import pandas as pd
from sklearn.svm import SVR
import numpy as np

# 创建预测结果文件夹
if not os.path.exists('预测结果'):
    os.makedirs('预测结果')

# 获取原数据文件夹中的所有csv文件
files = [f for f in os.listdir('原数据') if f.endswith('.csv')]

# 遍历每个文件进行预测
for file in files:
    # 读取数据
    data = pd.read_csv(f'原数据/{file}')
    
    # 使用全部数据训练 (1980-2021)
    X_train = data['year'].values.reshape(-1, 1)
    y_train = data['value'].values
    
    # 创建并训练SVM模型
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr.fit(X_train, y_train)
    
    # 预测2022-2050年数据
    future_years = np.arange(2022, 2051).reshape(-1, 1)
    y_pred = svr.predict(future_years)
    
    # 保存预测结果
    result = pd.DataFrame({
        'year': future_years.flatten(),
        'predicted_value': y_pred
    })
    result.to_csv(f'预测结果/result_{file}', index=False)

print("预测完成，结果已保存至'预测结果'文件夹")