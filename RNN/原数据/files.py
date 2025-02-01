import os

def list_files_in_directory():
    # 获取当前脚本所在的文件夹路径
    current_dir = os.getcwd()
    
    # 获取该文件夹下所有条目（包括文件和子目录）
    all_entries = os.listdir(current_dir)
    
    # 过滤出文件（排除子目录）
    files = [entry for entry in all_entries if os.path.isfile(os.path.join(current_dir, entry))]
    
    # 按字母顺序排序
    files_sorted = sorted(files)
    
    # 打印结果
    print(f"当前文件夹路径：{current_dir}")
    print("\n文件列表：")
    for idx, file in enumerate(files_sorted, 1):
        print(f"{idx}. {file}")

if __name__ == "__main__":
    list_files_in_directory()