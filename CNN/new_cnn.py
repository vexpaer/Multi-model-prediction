import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# 创建CNN模型
def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 处理单个文件
# ... existing code ...

def process_file(file_path):
    # 读取数据
    df = pd.read_csv(file_path)
    values = df['value'].values
    
    # 准备训练数据
    train_data = values[:31]  # 1980-2010年数据
    test_data = values[31:]   # 2011-2021年数据
    
    # 创建时间序列窗口
    def create_dataset(data, look_back=10):  # 将窗口大小改为10年
        X, y = [], []
        for i in range(len(data)-look_back):
            X.append(data[i:i+look_back])
            y.append(data[i+look_back])
        return np.array(X), np.array(y)
    
    # 使用全部31年数据训练
    X_train, y_train = create_dataset(values[:31])  # 1980-2010年
    # 使用2011-2021年数据测试
    X_test, y_test = create_dataset(values[21:])    # 2001-2021年
    
    # 调整数据形状以适应CNN
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # ... existing code ...
    
    # 训练模型
    model = create_cnn_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=100, verbose=0)
    
    # 预测
    predictions = model.predict(X_test)
    
    # 保存结果
    result_df = pd.DataFrame({
        'year': df['year'][-len(predictions):],
        'actual': y_test,
        'predicted': predictions.flatten()
    })
    return result_df

# 主程序
if __name__ == '__main__':
    # 创建结果文件夹
    os.makedirs('年龄结果', exist_ok=True)
    
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
    
    # 处理所有文件
    for file in files:
        file_path = os.path.join('原数据', file)
        result = process_file(file_path)
        output_path = os.path.join('年龄结果', f'result_{file}')
        result.to_csv(output_path, index=False)
        print(f'{file} 处理完成，结果已保存至 {output_path}')