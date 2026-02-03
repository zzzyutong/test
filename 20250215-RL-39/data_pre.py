#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

# 确保文件夹存在
if not os.path.exists("dataset"):
    os.makedirs("dataset")
if not os.path.exists("dataset/20241201pal_csv"):
    os.makedirs("dataset/20241201pal_csv")

# --------------------------
# 1. 生成原始的只有11列的csv
# --------------------------

data_11cols = []

# 假设日期为1到31号，根据实际数据情况调整
for day in range(1, 32):
    filename = f"dataset/20241201pal_csv/202412{day:02d}pal.csv"
    if not os.path.exists(filename):
        print(f"文件 {filename} 不存在，跳过该日期。")
        continue

    # 读取CSV文件，假设文件无header且数据从第二行开始
    data = pd.read_csv(filename, header=None, skiprows=1)
    # 取出第五列数据（索引4）
    column_data = data.iloc[:, 4].values

    # 按11个连续数字分组（不足11个则忽略）
    for i in range(0, len(column_data), 11):
        if i + 11 > len(column_data):
            break
        sample = column_data[i:i+11].tolist()
        data_11cols.append(sample)

# 保存11列原始数据
df_11 = pd.DataFrame(data_11cols)
df_11.to_csv('dataset/data_to_train_11.csv', index=False, header=False)
print("生成11列原始数据，保存为 'dataset/data_to_train_11.csv'")


# -------------------------------------------------------------
# 2. 根据提示对11列数据扩展10列
#    对于每一次扩展：
#      - 第一行：随机生成一个在本行[min, max]范围内的数，
#                添加20%的噪声得到 b，计算比例 p = b/（本行数之和）
#      - 其他行：计算本行之和 * p，添加5%的噪声作为 c
#    将生成的新列添加到数据右侧
#    合并后在指定位置插入全0列，
#    保存为 'dataset/data_to_train_original.csv'
# -------------------------------------------------------------

# 读取11列数据
df_11 = pd.read_csv('dataset/data_to_train_11.csv', header=None)

# 将新生成的列存放在列表中，最后水平拼接
new_cols = []  # 每个元素为一维 array，与 df_11 行数相同

# 重复10次扩展
for repeat in range(10):
    new_col = []  # 用于存放新生成列的数据
    # 获取第一行数据，转换为 float 数组
    first_row = df_11.iloc[0].values.astype(float)
    min_val = first_row.min()
    max_val = first_row.max()
    # 在[min_val, max_val]区间内随机生成一个数
    rand_val = np.random.uniform(min_val, max_val)
    # 添加20%噪声：这里的标准差取10% * (max_val - min_val)
    noise_b = np.random.normal(0, 0.1 * (max_val - min_val))
    b = rand_val + noise_b
    # 计算比例 p
    sum_first = first_row.sum()
    p = b / sum_first if sum_first != 0 else 0

    # 对于所有行分别计算新值
    for idx, row in df_11.iterrows():
        row_vals = row.values.astype(float)
        sum_row = row_vals.sum()
        if idx == 0:
            new_val = b
        else:
            c = sum_row * p
            # 添加5%的噪声：标准差取1% * abs(sum_row) * p
            noise_c = np.random.normal(0, 0.01 * abs(sum_row) * p)
            new_val = c + noise_c
        new_col.append(new_val)
    
    new_cols.append(new_col)

# 将10列新数据合并为 DataFrame
df_new = pd.DataFrame(np.array(new_cols).T)

# 将新的10列与原始11列数据横向合并
df_extended = pd.concat([df_11, df_new], axis=1)
# 此时 df_extended 共有 11 + 10 = 21 列

# ---------------------------
# 插入全0列到特定位置
# 指定位置（1-indexed）：2, 5, 6, 10, 11, 13, 14, 17, 19,
#                      22, 30, 32, 33, 34, 35, 36, 37, 38
# 注意：Python中第一列下标为0，因此对应1-indexed位置为：
#       2->位置1, 5->4, 6->5, 10->9, 11->10, 13->12, 14->13,
#       17->16, 19->18, 22->21, 30->29, 32->31, 33->32,
#       34->33, 35->34, 36->35, 37->36, 38->37
# 采用方法：构造最终39列（原21列+插入18列）的新DataFrame，
# 如果当前1-indexed列号在插入列表中则填充全0列，否则取 df_extended 中的下一列
# ---------------------------
insert_positions = {2, 5, 6, 10, 11, 13, 14, 17, 19, 22, 30, 32, 33, 34, 35, 36, 37, 38}
num_original_cols = df_extended.shape[1]  # 21列
num_inserts = len(insert_positions)         # 18列
total_cols = num_original_cols + num_inserts  # 最终 39 列
num_rows = df_extended.shape[0]

final_cols = []
orig_idx = 0
# 遍历1到total_cols（1-indexed）
for pos in range(1, total_cols + 1):
    if pos in insert_positions:
        # 构造一列全0数据
        final_cols.append(pd.Series(np.zeros(num_rows)))
    else:
        final_cols.append(df_extended.iloc[:, orig_idx])
        orig_idx += 1

# 合并成最终DataFrame，列顺序与要求一致
df_final = pd.concat(final_cols, axis=1)

# 保存合并扩展、并插入全0列后的数据
df_final.to_csv('dataset/data_to_train_original.csv', index=False, header=False)
print("生成扩展数据（原始11列+新生成的10列，再插入全0列），保存为 'dataset/data_to_train_original.csv'")


# -------------------------------------------------------------
# 3. 归一化 'dataset/data_to_train_original.csv' 到 (-1, 1)
# -------------------------------------------------------------

# 读取扩展后的数据（包含插入的0列）
df_original = pd.read_csv('dataset/data_to_train_original.csv', header=None)

# 计算全局最小值和最大值（所有元素）
global_min = df_original.min().min()
global_max = df_original.max().max()

if global_max == global_min:
    df_norm = df_original * 0.0
else:
    # 先归一化到[0,1]
    df_norm = (df_original - global_min) / (global_max - global_min)
    # 转换到[-1,1]
    df_norm = df_norm * 2 - 1

df_norm.to_csv('dataset/data_to_train.csv', index=False, header=False)
print("归一化数据到(-1, 1)，保存为 'dataset/data_to_train.csv'")
