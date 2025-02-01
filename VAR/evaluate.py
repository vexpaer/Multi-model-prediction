import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(actual, predicted):
    # 计算RMSE
    rmse = mean_squared_error(actual, predicted, squared=False)
    
    # 计算MAE
    mae = mean_absolute_error(actual, predicted)
    
    # 计算MAPE
    mape = 100 * (abs(actual - predicted) / actual).mean()
    
    # 计算R²
    r2 = r2_score(actual, predicted)
    
    return rmse, mae, mape, r2

def process_folder(folder_path):
    results = {}
    
    # 遍历文件夹中的所有CSV文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 计算指标
            rmse, mae, mape, r2 = calculate_metrics(df['actual'], df['predicted'])
            
            # 存储结果
            results[filename] = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R²': r2
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
    print(f"  R²: {metrics['R²']:.4f}")
    print()