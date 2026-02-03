import pandas as pd
import numpy as np
import os

def generate_solar_monte_carlo_v2():
    train_file = 'dataset/data_to_train.csv'
    pv_file = 'dataset/PV_Live Historical Results.csv'
    output_file = 'dataset/solar.csv'

    # -----------------------------------------------------------
    # 1. 从 data_to_train.csv 识别负荷节点
    # -----------------------------------------------------------
    if not os.path.exists(train_file):
        print(f"错误：找不到文件 {train_file}")
        return

    try:
        # 读取训练数据（假设无表头）
        df_train = pd.read_csv(train_file, header=None)
        print(f"读取 {train_file} 成功，形状: {df_train.shape}")

        # 逻辑：计算每一列的标准差。
        # 如果标准差 > 0 (或非常小的阈值)，说明该节点数据有变化，判定为负荷节点。
        # 如果标准差 == 0 (通常是归一化后的 -1.0)，说明该节点无负荷。
        std_values = df_train.std()
        load_indices = std_values[std_values > 1e-6].index.tolist()
        
        print(f"识别到 {len(load_indices)} 个负荷节点 (Column Indices): {load_indices}")
        
    except Exception as e:
        print(f"分析训练数据时出错: {e}")
        return

    # -----------------------------------------------------------
    # 2. 读取并处理 PV Live 数据
    # -----------------------------------------------------------
    if not os.path.exists(pv_file):
        print(f"错误：找不到文件 {pv_file}")
        return

    try:
        df_pv = pd.read_csv(pv_file)
        
        # 按照之前的逻辑进行切片
        # Excel行号 1537 -> Index 1535 (Header占1行)
        # Excel行号 49   -> Index 47
        # 步长 -2
        # 切片 end 是不包含的，为了包含 47，由于步长为负，end 设为 46
        start_idx = 1535
        end_idx = 46 
        
        pv_slice = df_pv.iloc[start_idx : end_idx : -2, 2] # 取第3列
        
        # 转为 numpy 数组并处理空值
        raw_data = pd.to_numeric(pv_slice, errors='coerce').fillna(0).values
        print(f"提取光伏基准数据点数: {len(raw_data)}")
        
    except Exception as e:
        print(f"读取光伏数据时出错: {e}")
        return

    # -----------------------------------------------------------
    # 3. 归一化基准曲线到 [0, 0.1]
    # -----------------------------------------------------------
    min_val = raw_data.min()
    max_val = raw_data.max()
    
    if max_val > min_val:
        base_curve = (raw_data - min_val) / (max_val - min_val) * 0.1
    else:
        base_curve = np.zeros_like(raw_data)
        
    print(f"基准曲线归一化完成: Range [{base_curve.min():.4f}, {base_curve.max():.4f}]")

    # -----------------------------------------------------------
    # 4. 蒙特卡洛模拟生成最终矩阵
    # -----------------------------------------------------------
    num_rows = len(base_curve)
    num_cols = df_train.shape[1] # 保持与 data_to_train 列数一致 (39)
    
    # 初始化全 0 矩阵
    solar_matrix = np.zeros((num_rows, num_cols))
    
    print("开始生成随机波动数据...")
    
    # 仅遍历被识别为负荷的列
    for col_idx in load_indices:
        # 生成随机波动: 范围 [-0.05, 0.05]
        fluctuation = np.random.uniform(-0.05, 0.05, size=num_rows)
        
        # 叠加波动
        simulated = base_curve + fluctuation
        
        # 截断：最小不能低于 0
        simulated = np.maximum(simulated, 0)
        
        # 赋值
        solar_matrix[:, col_idx] = simulated

    # -----------------------------------------------------------
    # 5. 保存结果
    # -----------------------------------------------------------
    # 格式：无表头，无索引，与 data_to_train.csv 一致
    output_df = pd.DataFrame(solar_matrix)
    output_df.to_csv(output_file, index=False, header=False)
    
    print("------------------------------------------------")
    print(f"成功生成 {output_file}")
    print(f"矩阵形状: {output_df.shape}")
    print(f"非负荷节点值已设为 0")
    print("------------------------------------------------")

if __name__ == "__main__":
    generate_solar_monte_carlo_v2()