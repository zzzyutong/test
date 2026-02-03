# 定义文件路径
file_path = r'C:\Users\50220\miniconda3\envs\rvc\lib\site-packages\pypower\case39.py'
mod_file_path = r'C:\Users\50220\miniconda3\envs\rvc\lib\site-packages\pypower\case39-mod.py'

# 定义需要修改的行号区间
start_line = 96  # 起始行号
end_line = 134    # 结束行号

# 打开文件进行读取
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 创建一个新文件来保存修改后的内容
with open(mod_file_path, 'w', encoding='utf-8') as file:
    for i, line in enumerate(lines, start=1):  # 行号从1开始
        if start_line <= i <= end_line:
            # 按逗号分割行内容
            parts = line.rstrip('\n').split(',')
            if len(parts) >= 3:  # 确保该行至少有三列
                # 修改第三列的内容，行号从 0 开始，所以用 (i - start_line) 来确定第三列的值
                parts[2] = f' L[{i - start_line}]'
            # 写回修改后的行，不添加额外空格，并保留原有的换行符
            file.write(','.join(parts) + '\n')
        else:
            # 不需要修改的行，直接写回，不添加空格
            file.write(line)

print(f"文件的第 {start_line} 行到第 {end_line} 行已成功修改！")
