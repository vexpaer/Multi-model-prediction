import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 创建结果文件夹
if not os.path.exists('年龄结果'):
    os.makedirs('年龄结果')

# 文件列表
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
    data = pd.read_csv(f'原数据/{file}')
    
    # 划分训练集和测试集
    train = data[(data['year'] >= 1980) & (data['year'] <= 2010)]
    test = data[(data['year'] >= 2011) & (data['year'] <= 2021)]
    
    # 准备训练数据
    X_train = train['year'].values.reshape(-1, 1)
    y_train = train['value'].values
    
    # 初始化模型
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 递归预测
    predictions = []
    last_year = 2010
    last_value = train[train['year'] == 2010]['value'].values[0]
    
    for year in range(2011, 2022):
        pred = model.predict([[year]])
        predictions.append({'year': year, 'actual': test[test['year'] == year]['value'].values[0], 
                          'predicted': pred[0]})
        last_year = year
        last_value = pred[0]
    
    # 保存结果
    result_df = pd.DataFrame(predictions)
    result_df.to_csv(f'年龄结果/result_{file}', index=False, float_format='%.8f')

print("所有文件处理完成！")