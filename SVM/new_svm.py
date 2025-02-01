import os
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
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
    
    # 准备训练数据 (1980-2010)
    train_data = data.iloc[:31]  # 前31行是1980-2010年数据
    X_train = train_data['year'].values.reshape(-1, 1)
    y_train = train_data['value'].values
    
    # 准备测试数据 (2011-2021)
    test_data = data.iloc[31:]  # 后11行是2011-2021年数据
    X_test = test_data['year'].values.reshape(-1, 1)
    y_test = test_data['value'].values
    
    # 创建并训练SVM模型
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr.fit(X_train, y_train)
    
    # 预测
    y_pred = svr.predict(X_test)
    
    # 保存结果
    result = pd.DataFrame({
        'year': test_data['year'],
        'actual_value': y_test,
        'predicted_value': y_pred
    })
    result.to_csv(f'年龄结果/result_{file}', index=False)