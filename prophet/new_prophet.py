import pandas as pd
from prophet import Prophet
import os

# 创建结果文件夹
if not os.path.exists('年龄结果'):
    os.makedirs('年龄结果')

# 定义文件列表
files = [
    '10-14_years.csv', '15-19_years.csv', '20-24_years.csv',
    '25-29_years.csv', '30-34_years.csv', '35-39_years.csv',
    '40-44_years.csv', '45-49_years.csv', '5-9_years.csv',
    '50-54_years.csv', '55-59_years.csv', '60-64_years.csv',
    '65-69_years.csv', '70-74_years.csv', '75-79_years.csv',
    '80-84_years.csv', '85-89_years.csv', '90-94_years.csv',
    '95plus_years.csv', 'less_5_years.csv'
]

for file in files:
    # 读取数据
    df = pd.read_csv(f'原数据/{file}')
    
    # 准备Prophet所需格式
    df = df.rename(columns={'year': 'ds', 'value': 'y'})
    
    # 划分训练集和预测集
    train = df[df['ds'] <= 2010]
    future = pd.DataFrame({'ds': range(2011, 2022)})
    
    # 训练模型
    model = Prophet()
    model.fit(train)
    
    # 预测
    forecast = model.predict(future)
    
    # 保存结果
    result = forecast[['ds', 'yhat']].rename(columns={'ds': 'year', 'yhat': 'value'})
    result.to_csv(f'年龄结果/result_{file}', index=False, float_format='%.8f')