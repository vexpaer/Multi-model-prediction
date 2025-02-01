import os
import pandas as pd
from statsmodels.tsa.api import VAR

# 创建预测结果文件夹
if not os.path.exists('预测结果'):
    os.makedirs('预测结果')

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
    
    # 使用1980-2021年数据训练
    train_data = data[(data['year'] >= 1980) & (data['year'] <= 2021)]
    
    # 创建VAR模型
    model = VAR(train_data[['year', 'value']])
    results = model.fit(maxlags=2, ic='aic')
    
    # 预测2022-2050年数据
    lag_order = results.k_ar
    forecast_input = train_data.values[-lag_order:]
    forecast = results.forecast(y=forecast_input, steps=29)  # 29 steps对应2022-2050年
    
    # 创建预测年份
    forecast_years = range(2022, 2051)
    
    # 保存预测结果
    result_df = pd.DataFrame({
        'year': forecast_years,
        'predicted': forecast[:, 1]
    })
    result_df.to_csv(f'预测结果/result_{file}', index=False, float_format='%.8f')