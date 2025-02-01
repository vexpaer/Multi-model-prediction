import os
import pandas as pd
from prophet import Prophet

# 创建结果文件夹
if not os.path.exists('预测结果'):
    os.makedirs('预测结果')

# 获取原数据文件夹中的所有文件
files = [f for f in os.listdir('原数据') if f.endswith('.csv')]

for file in files:
    # 读取数据
    df = pd.read_csv(f'原数据/{file}')
    
    # 准备Prophet所需格式
    df = df.rename(columns={'year': 'ds', 'value': 'y'})
    
    # 划分训练集和预测集
    train = df[(df['ds'] >= 1980) & (df['ds'] <= 2021)]
    future = pd.DataFrame({'ds': range(2022, 2051)})
    
    # 训练模型
    model = Prophet()
    model.fit(train)
    
    # 预测
    forecast = model.predict(future)
    
    # 保存结果
    result = forecast[['ds', 'yhat']].rename(columns={'ds': 'year', 'yhat': 'value'})
    result.to_csv(f'预测结果/result_{file}', index=False, float_format='%.8f')

print("预测完成，结果已保存至'预测结果'文件夹")