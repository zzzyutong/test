import pandas as pd
import json

# 1. 数据清洗（去重、补全、排序）
load_df = pd.read_csv("dataset/data_to_train.csv")
load_df = load_df.drop_duplicates()  # 剔除重复行
load_df["timestamp"] = pd.to_datetime(load_df["timestamp"])  # 标准化时间戳
load_df = load_df.sort_values("timestamp").fillna(method="ffill")  # 填充缺失值

# 2. 5分钟→1小时采样（间隔12条）
hourly_load = load_df.iloc[::12].reset_index(drop=True)

# 3. 构造微调样本（输入：历史序列+场景；输出：目标负荷）
fine_tune_samples = []
for i in range(3, len(hourly_load)):
    # 提取历史3个时刻的负荷特征
    history = hourly_load.iloc[i-3:i][["timestamp", "total_P_load", "bus18_U"]].to_dict("records")
    history_str = "; ".join([f"{h['timestamp']}总负荷{h['total_P_load']}MW、节点18电压{h['bus18_U']}pu" for h in history])
    
    # 构建Prompt（贴合电力场景）
    prompt = f"已知case39电力系统的历史运行数据：[{history_str}]，当前时间为{hourly_load.iloc[i]['timestamp']}（工作日早高峰），求该时刻系统总有功负荷值"
    # 构建Answer（精准数值+单位）
    answer = f"{hourly_load.iloc[i]['total_P_load']}MW"
    
    fine_tune_samples.append({"prompt": prompt, "answer": answer})

# 4. 保存为JSONL（微调专用格式）
with open("data/load_pred.jsonl", "w", encoding="utf-8") as f:
    for sample in fine_tune_samples:
        json.dump(sample, f, ensure_ascii=False)
        f.write("\n")