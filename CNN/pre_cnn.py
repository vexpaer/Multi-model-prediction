import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# 创建CNN模型
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def process_file(file_path):
    # 读取数据
    df = pd.read_csv(file_path)
    values = df['value'].values
    
    # 创建时间序列窗口
    def create_dataset(data, look_back=10):
        X, y = [], []
        for i in range(len(data)-look_back):
            X.append(data[i:i+look_back])
            y.append(data[i+look_back])
        return np.array(X), np.array(y)
    
    # 使用全部1980-2021年数据训练
    X_train, y_train = create_dataset(values)  # 全部数据用于训练
    
    # 准备预测数据（最后10年数据用于预测）
    last_window = values[-10:]
    future_predictions = []
    
    # 调整数据形状以适应CNN
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    # 训练模型
    model = create_cnn_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=100, verbose=0)
    
    # 预测2022-2050年数据
    current_window = last_window.copy()
    for year in range(2022, 2051):
        # 准备输入数据
        input_data = current_window.reshape((1, 10, 1))
        # 预测下一年
        prediction = model.predict(input_data, verbose=0)[0][0]
        future_predictions.append(prediction)
        # 更新窗口
        current_window = np.append(current_window[1:], prediction)
    
    # 保存结果
    result_df = pd.DataFrame({
        'year': range(2022, 2051),
        'predicted': future_predictions
    })
    return result_df

# 主程序
if __name__ == '__main__':
    # 创建结果文件夹
    os.makedirs('预测结果', exist_ok=True)
    
    # 获取所有数据文件
    data_dir = '原数据'
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # 处理所有文件
    for file in files:
        file_path = os.path.join(data_dir, file)
        result = process_file(file_path)
        output_path = os.path.join('预测结果', f'result_{file}')
        result.to_csv(output_path, index=False)
        print(f'{file} 处理完成，结果已保存至 {output_path}')