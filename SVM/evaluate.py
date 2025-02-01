import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(df):
    # 计算各项指标
    rmse = mean_squared_error(df['actual_value'], df['predicted_value'], squared=False)
    mae = mean_absolute_error(df['actual_value'], df['predicted_value'])
    mape = (abs((df['actual_value'] - df['predicted_value']) / df['actual_value'])).mean() * 100
    r2 = r2_score(df['actual_value'], df['predicted_value'])
    
    return rmse, mae, mape, r2

def process_folder(folder_path):
    # 遍历文件夹中的所有CSV文件
    results = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # 计算指标
            rmse, mae, mape, r2 = calculate_metrics(df)
            
            # 存储结果
            results[filename] = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2
            }
    
    return results

# 使用示例
folder_path = '年龄结果'
metrics_results = process_folder(folder_path)

# 打印结果
for filename, metrics in metrics_results.items():
    print(f"文件: {filename}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.4f}%")
    print(f"  R²: {metrics['R2']:.4f}")
    print()