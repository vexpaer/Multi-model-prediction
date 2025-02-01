import os
import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error

# 创建结果文件夹
if not os.path.exists('年龄结果'):
    os.makedirs('年龄结果')

# 定义文件列表
files = [
    '10-14_years.csv', '15-19_years.csv', '20-24_years.csv', '25-29_years.csv',
    '30-34_years.csv', '35-39_years.csv', '40-44_years.csv', '45-49_years.csv',
    '5-9_years.csv', '50-54_years.csv', '55-59_years.csv', '60-64_years.csv',
    '65-69_years.csv', '70-74_years.csv', '75-79_years.csv', '80-84_years.csv',
    '85-89_years.csv', '90-94_years.csv', '95plus_years.csv', 'less_5_years.csv'
]

# 处理每个文件
for file in files:
    # 读取数据
    data = pd.read_csv(f'原数据/{file}')
    
    # 准备数据
    train_data = data[(data['year'] >= 1980) & (data['year'] <= 2010)]
    test_data = data[(data['year'] >= 2011) & (data['year'] <= 2021)]
    
    # 创建VAR模型
    model = VAR(train_data[['year', 'value']])
    results = model.fit(maxlags=2, ic='aic')
    
    # 预测
    lag_order = results.k_ar
    forecast_input = train_data.values[-lag_order:]
    forecast = results.forecast(y=forecast_input, steps=11)
    
    # 保存结果
    result_df = pd.DataFrame({
        'year': test_data['year'],
        'actual': test_data['value'],
        'predicted': forecast[:, 1]
    })
    result_df.to_csv(f'年龄结果/result_{file}', index=False, float_format='%.8f')