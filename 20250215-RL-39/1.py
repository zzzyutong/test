import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_load
import train_power_flow
from config_loader import get_train_base_load, get_train_load_rate

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

device = torch.device("cpu")
print(f"Simulation running on device: {device}")

os.makedirs("power_flow", exist_ok=True)
os.makedirs("results_plot", exist_ok=True)

def tensor_to_mw(data_norm, base_load, load_rate):
    """
    辅助函数：将归一化数据 [-1, 1] 映射回 MW 值
    公式与 train_power_flow.py 中保持一致
    """
    top_load = load_rate * base_load
    # 1. [-1, 1] -> [0, 1]
    data_01 = (data_norm + 1) / 2
    # 2. [0, 1] -> [0, top_load] -> +1 -> MW
    data_mw = data_01 * top_load + 1
    return data_mw

def run_simulation():
    # 1. 加载负荷数据
    dataloader = data_load.load_data_from_csv()
    
    # 2. 加载光伏数据
    solar_file = 'dataset/solar.csv'
    if not os.path.exists(solar_file):
        print(f"错误：找不到 {solar_file}")
        return
    
    # 读取原始光伏数据 (numpy array)
    solar_raw_df = pd.read_csv(solar_file, header=None)
    solar_raw = solar_raw_df.values.astype('float32') # shape: (N_solar, 39)
    print(f"负荷数据加载完成。光伏数据加载完成，长度: {len(solar_raw)}")

    # 获取配置参数用于转换
    base_load = get_train_base_load()
    load_rate = get_train_load_rate()

    # --- 初始化历史数据容器 ---
    history_gen_pg = []       # 发电机出力
    history_load_total = []   # 原始总负荷 (Load)
    history_solar_total = []  # 光伏总出力 (Solar)
    
    # 用于动态存储实际使用的光伏数据（用于处理长度不足的回溯）
    used_solar_history = [] 

    print(f"开始仿真，共 {len(dataloader)} 组数据...")

    # 3. 遍历数据集
    for i, (real_data_samples,) in enumerate(dataloader):
        # real_data_samples: [1, 39], 归一化的原始负荷
        load_norm = real_data_samples.to(device)[0] # shape [39]
        
        # --- 处理光伏数据 (补全逻辑) ---
        if i < len(solar_raw):
            # 如果还在光伏数据长度范围内，直接取
            current_solar_np = solar_raw[i]
        else:
            # 如果超出范围，取24个之前的对应数据
            # 注意：从 used_solar_history 中取，确保是连贯的
            lookback_idx = i - 24
            if lookback_idx < 0:
                # 极端情况：如果 solar 数据本身小于 24 行，取模循环
                lookback_idx = i % len(solar_raw)
                current_solar_np = solar_raw[lookback_idx]
            else:
                current_solar_np = used_solar_history[lookback_idx]
        
        # 存入历史，供未来回溯使用
        used_solar_history.append(current_solar_np)
        
        # 转为 Tensor
        solar_norm = torch.from_numpy(current_solar_np).to(device)

        # --- 核心计算 ---
        
        # 1. 计算净负荷 (归一化空间)
        # Net = Load - Solar
        net_load_norm = load_norm - solar_norm
        
        # 2. 调用潮流计算 (传入净负荷)
        # loadflow 内部会将 net_load_norm 转化为真实 MW 并计算
        gen_pg, _ = train_power_flow.loadflow(net_load_norm, episode="sim", i=i, flag="test")
        
        # 3. 计算用于绘图的真实 MW 值
        # 将 Tensor 转回 numpy 计算总和
        load_norm_np = load_norm.detach().cpu().numpy()
        net_load_norm_np = net_load_norm.detach().cpu().numpy()
        
        # 还原原始负荷 (MW)
        load_mw = tensor_to_mw(load_norm_np, base_load, load_rate)
        # 还原净负荷 (MW)
        net_load_mw = tensor_to_mw(net_load_norm_np, base_load, load_rate)
        
        # 计算光伏出力 (MW) = 原始负荷 - 净负荷
        # 注意：这里不能直接用 tensor_to_mw(solar_norm)，因为映射公式包含常数项 +1，直接相减抵消才是 ΔP
        solar_mw = load_mw - net_load_mw
        
        # 记录总量 (Sum of all 39 buses)
        history_load_total.append(np.sum(load_mw))
        history_solar_total.append(np.sum(solar_mw))
        history_gen_pg.append(gen_pg)
        
        if (i + 1) % 50 == 0:
            print(f"已完成 {i + 1} 步仿真...")

    print("仿真完成，正在生成图表...")

    # 4. 转换数据与绘图
    np_gen_pg = np.array(history_gen_pg)
    np_load_total = np.array(history_load_total)
    np_solar_total = np.array(history_solar_total)
    
    time_steps = range(len(np_load_total))

    # --- 图表 1: 原始负荷 vs 光伏曲线 ---
    plt.figure(figsize=(12, 6))
    
    # 绘制原始负荷 (左轴)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('时间步 (Time Step)')
    ax1.set_ylabel('系统总负荷 (MW)', color=color)
    ax1.plot(time_steps, np_load_total, label='原始总负荷 (Original Load)', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # 绘制光伏 (右轴，因为光伏数值可能较小，双轴显示更清晰)
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('系统总光伏出力 (MW)', color=color)  
    ax2.plot(time_steps, np_solar_total, label='光伏出力 (Solar Generation)', color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("原始负荷与实时光伏出力曲线")
    fig.tight_layout()  
    plt.savefig("results_plot/load_and_solar.png", dpi=300)
    plt.close()

    # --- 图表 2: 发电机出力曲线 ---
    plt.figure(figsize=(12, 6))
    num_gens = np_gen_pg.shape[1]
    gen_bus_ids = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39] 
    
    for g_idx in range(num_gens):
        plt.plot(time_steps, np_gen_pg[:, g_idx], label=f'Gen Bus {gen_bus_ids[g_idx]}')
    
    plt.title("IEEE 39节点系统 - 发电机实时出力 (PG)")
    plt.xlabel("时间步 (Time Step)")
    plt.ylabel("有功出力 (MW)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results_plot/generator_output.png", dpi=300)
    plt.close()

    print("绘图完成：")
    print("1. results_plot/load_and_solar.png (负荷与光伏)")
    print("2. results_plot/generator_output.png (发电机出力)")

if __name__ == "__main__":
    run_simulation()