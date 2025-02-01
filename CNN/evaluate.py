import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(actual, predicted):
    # 计算RMSE
    rmse = mean_squared_error(actual, predicted, squared=False)
    
    # 计算MAE
    mae = mean_absolute_error(actual, predicted)
    
    # 计算MAPE
    mape = (abs((actual - predicted) / actual)).mean() * 100
    
    # 计算R方
    r2 = r2_score(actual, predicted)
    
    return rmse, mae, mape, r2

def process_folder(folder_path):
    results = []
    
    # 遍历文件夹中的所有csv文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            
            # 读取csv文件
            df = pd.read_csv(file_path)
            
            # 计算指标
            rmse, mae, mape, r2 = calculate_metrics(df['actual'], df['predicted'])
            
            # 保存结果
            results.append({
                'filename': filename,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2
            })
    
    # 将结果保存到DataFrame中
    result_df = pd.DataFrame(results)
    return result_df

# 使用示例
folder_path = '年龄结果'
metrics_df = process_folder(folder_path)
print(metrics_df)
# 如果需要保存结果到csv
# metrics_df.to_csv('metrics_results.csv', index=False)