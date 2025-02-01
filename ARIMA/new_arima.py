import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# 创建结果文件夹
os.makedirs('年龄结果', exist_ok=True)

# 文件列表
files = [
    '10-14_years.csv', '15-19_years.csv', '20-24_years.csv', '25-29_years.csv',
    '30-34_years.csv', '35-39_years.csv', '40-44_years.csv', '45-49_years.csv',
    '5-9_years.csv', '50-54_years.csv', '55-59_years.csv', '60-64_years.csv',
    '65-69_years.csv', '70-74_years.csv', '75-79_years.csv', '80-84_years.csv',
    '85-89_years.csv', '90-94_years.csv', '95plus_years.csv', 'less_5_years.csv'
]

for file in files:
    # 读取数据
    data = pd.read_csv(f'原数据/{file}')
    
    # 训练数据 (1980-2010)
    train = data.iloc[:31]['value']
    
    # 创建并训练ARIMA模型
    model = ARIMA(train, order=(1,1,1))
    model_fit = model.fit()
    
    # 预测2011-2021年数据
    forecast = model_fit.forecast(steps=11)
    
    # 创建结果DataFrame，只包含预测年份和对应实际值
    result = pd.DataFrame({
        'year': data['year'].iloc[31:],  # 只取2011-2021年
        'actual': data['value'].iloc[31:],  # 只取2011-2021年实际值
        'predicted': forecast  # 预测值
    })
    
    # 保存结果
    result.to_csv(f'年龄结果/result_{file}', index=False)