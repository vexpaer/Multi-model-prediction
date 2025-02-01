import os
import pandas as pd

# 定义文件夹路径
original_folder = '原数据'
result_folder = '年龄结果'
output_folder = '合并结果'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取两个文件夹下的所有CSV文件
original_files = [f for f in os.listdir(original_folder) if f.endswith('.csv')]
result_files = [f for f in os.listdir(result_folder) if f.endswith('.csv')]

# 确保文件数量一致
if len(original_files) != len(result_files):
    print("错误：原数据和结果文件数量不匹配")
    exit()

# 处理每个文件
for original_file, result_file in zip(original_files, result_files):
    # 读取原数据
    original_df = pd.read_csv(os.path.join(original_folder, original_file))
    # 读取预测结果
    result_df = pd.read_csv(os.path.join(result_folder, result_file))
    
    # 过滤2011-2021年的数据
    original_df = original_df[(original_df['year'] >= 2011) & (original_df['year'] <= 2021)]
    result_df = result_df[(result_df['year'] >= 2011) & (result_df['year'] <= 2021)]
    
    # 合并数据
    merged_df = original_df.merge(result_df, on='year', suffixes=('_original', '_predicted'))
    
    # 保存合并后的文件
    output_path = os.path.join(output_folder, original_file)
    merged_df.to_csv(output_path, index=False)
    print(f"已保存合并文件: {output_path}")

print("所有文件合并完成！")