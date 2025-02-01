import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(df):
    # 计算各项指标
    rmse = mean_squared_error(df['value_original'], df['value_predicted'], squared=False)
    mae = mean_absolute_error(df['value_original'], df['value_predicted'])
    mape = (abs((df['value_original'] - df['value_predicted']) / df['value_original'])).mean() * 100
    r2 = r2_score(df['value_original'], df['value_predicted'])
    return rmse, mae, mape, r2

def process_folder(folder_path):
    results = []
    
    # 遍历文件夹中的所有CSV文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # 计算指标
            rmse, mae, mape, r2 = calculate_metrics(df)
            
            # 保存结果
            results.append({
                'filename': filename,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2
            })
    
    # 将结果保存到DataFrame
    result_df = pd.DataFrame(results)
    return result_df

# 使用示例
folder_path = '合并结果'
metrics_df = process_folder(folder_path)

# 打印结果
print(metrics_df)

# 保存结果到CSV
metrics_df.to_csv('模型评估指标.csv', index=False)