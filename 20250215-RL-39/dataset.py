import pandas as pd
import numpy as np
import os
import json

# 1. 配置加载与参数初始化
with open('config/train_config.json', 'r') as f:
    config = json.load(f)
base_load = config.get("train_base_load", 900)
load_rate = config.get("train_load_rate", 1)
top_load = base_load * load_rate

def to_mw(data):
    """根据脚本1.py的逻辑还原原始MW值"""
    return ((data + 1) / 2) * top_load + 1

def extract_all_gens(file_path):
    """从调度结果中提取10台发电机的各自出力"""
    gen_pgs = [np.nan] * 10 # IEEE 39节点系统通常有10台发电机
    if not os.path.exists(file_path):
        return gen_pgs
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            is_gen = False
            idx = 0
            for line in lines:
                row = line.strip()
                if row == "gen":
                    is_gen = True
                    continue
                if is_gen:
                    if row == "" or row == "branch" or row == "gencost": break
                    if "Bus ID" in row: continue
                    parts = row.split(',')
                    if len(parts) > 1 and idx < 10:
                        gen_pgs[idx] = float(parts[1]) # 提取第2列 Pg
                        idx += 1
        return gen_pgs
    except:
        return [np.nan] * 10

# 2. 处理负荷与新能源 (保持39列)
print("正在还原39个节点的负荷与新能源数据...")
load_raw = pd.read_csv('dataset/data_to_train.csv', header=None)
load_node_mw = to_mw(load_raw).iloc[::12].reset_index(drop=True) # 采样到小时

solar_raw = pd.read_csv('dataset/solar.csv', header=None)
solar_node_mw = to_mw(solar_raw).reset_index(drop=True)

# 3. 提取所有发电机出力
print("正在提取各机组调度结果...")
all_gen_data = []
for i in range(len(load_node_mw)):
    f_path = f"power_flow/test_sim_{i}.csv"
    all_gen_data.append(extract_all_gens(f_path))
gen_df = pd.DataFrame(all_gen_data, columns=[f'Gen{i+1}_Pg' for i in range(10)])

# 4. 构造节点级对齐数据集
print("正在进行全节点时刻对齐...")
min_len = min(len(load_node_mw), len(solar_node_mw), len(gen_df))
final_rows = []

for t in range(min_len - 1):
    # 构造当前时刻 t 的特征字典
    row = {"hour_index": t}
    
    # t 时刻数据
    for n in range(39):
        row[f"L{n+1}_t"] = round(load_node_mw.iloc[t, n], 2)
        row[f"S{n+1}_t"] = round(solar_node_mw.iloc[t, n], 2)
    for g in range(10):
        row[f"G{g+1}_t"] = round(gen_df.iloc[t, g], 2)
        
    # t+1 时刻预测/结果数据
    for n in range(39):
        row[f"L{n+1}_t+1_pre"] = round(load_node_mw.iloc[t+1, n], 2)
        row[f"S{n+1}_t+1_pre"] = round(solar_node_mw.iloc[t+1, n], 2)
    for g in range(10):
        row[f"G{g+1}_t+1_res"] = round(gen_df.iloc[t+1, g], 2)
        
    final_rows.append(row)

# 5. 保存结果
dataset_node_level = pd.DataFrame(final_rows)
# 剔除包含空值（调度失败或文件缺失）的行
dataset_node_level.dropna(inplace=True)
dataset_node_level.to_csv('node_level_power_data.csv', index=False)
print(f"✅ 处理完成！已生成 39节点+10机组 的对齐数据，共 {len(dataset_node_level)} 行，保存至 node_level_power_data.csv")