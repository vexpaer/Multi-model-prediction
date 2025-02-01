import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('原数据.csv')

# 获取所有唯一的年龄段
age_groups = df['age'].unique()

# 按年龄段分组并保存为单独的CSV文件
for age in age_groups:
    # 过滤出当前年龄段的数据
    age_df = df[df['age'] == age]
    
    # 只保留year和val列
    age_df = age_df[['year', 'value']]
    
    # 生成文件名（替换特殊字符）
    filename = f"{age.replace('<', 'less_').replace('+', 'plus').replace(' ', '_')}.csv"
    
    # 保存为CSV文件
    age_df.to_csv(filename, index=False)