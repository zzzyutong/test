import numpy as np
import csv
import os
import torch
from pypower.api import ppoption, runopf, case39
from pypower.idx_bus import PD, BUS_I  # 引入总线索引
from pypower.idx_gen import PG, GEN_BUS # 引入发电机索引
from config_loader import get_train_base_load
from config_loader import get_train_load_rate

# 定义发电机和线路限制
limit_gen = np.array([1040, 646, 725, 652, 508, 687, 580, 564, 865, 1100])
limit_branch = 100.0
limit_voltage = 70

def save_result(result, filename, decimal_places=2):
    formatted_results = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            formatted_results[key] = np.round(value, decimal_places)
        elif isinstance(value, (float, int)):
            formatted_results[key] = round(value, decimal_places)
        else:
            formatted_results[key] = value

    # 定义每类数据的列标题 (保持原有逻辑)
    column_titles = {
        'bus': ['Bus ID', 'Type', 'Pd', 'Qd', 'Gs', 'Bs', 'Area', 'Vm', 'Va', 'BaseKV', 'Zone', 'Vmax', 'Vmin', 'Lam_P', 'Lam_Q', 'Mu_Vmax', 'Mu_Vmin'],
        'gen': ['Bus ID', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'Status', 'Pmax', 'Pmin', 'Pc1', 'Pc2', 'Qc1min', 'Qc1max', 'Qc2min', 'Qc2max', 'Ramp_agc', 'Ramp_10', 'Ramp_30', 'Ramp_q', 'APF', 'Mu_Pmax', 'Mu_Pmin', 'Mu_Qmax', 'Mu_Qmin'],
        'branch': ['From Bus', 'To Bus', 'R', 'X', 'B', 'RateA', 'RateB', 'RateC', 'Ratio', 'Angle', 'Status', 'Angmin', 'Angmax', 'Pf', 'Qf', 'Pt', 'Qt', 'Mu_Sf', 'Mu_St', 'Mu_angmin', 'Mu_angmax'],
        'gencost': ['Model', 'Startup', 'Shutdown', 'N', 'Cost Coefficients'],
    }

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for key, value in formatted_results.items():
            writer.writerow([key])
            if key in column_titles:
                writer.writerow(column_titles[key])
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    writer.writerow(value.tolist())
                elif value.ndim == 2:
                    for row in value:
                        writer.writerow(row.tolist())
            else:
                writer.writerow([value])

def loadflow(load_data, episode, i, flag):
    # 1. 获取配置参数
    base_load = get_train_base_load()
    load_rate = get_train_load_rate()
    
    if flag in ['test', 'train_test', 'test1', 'test2', 'test3', 'simulation']:
        top_load = load_rate * base_load
    else:
        top_load = load_rate * base_load

    # 2. 数据反归一化与映射
    if isinstance(load_data, torch.Tensor):
        load_data = load_data.cpu()
    
    filename = f"power_flow/{flag}_{episode}_{i}.csv"
    if os.path.exists(filename):
        os.remove(filename)

    # 映射逻辑
    load_data = (load_data + 1) / 2
    load_data = load_data * top_load + 1
    
    if isinstance(load_data, torch.Tensor):
        load_data = load_data.detach().numpy()
        
    load_data_original = np.round(load_data, decimals=1)
    
    # 3. 运行 OPF
    data = case39(load_data_original)
    ppopt = ppoption(OPF_MAX_IT=1000, OUT_GEN=0, VERBOSE=0)
    result = runopf(data, ppopt, fname=filename)
    
    # 保存CSV (如果不需要保存文件可注释掉下面这行，提高速度)
    save_result(result, filename, decimal_places=2)

    # --- 新增：提取关键数据用于绘图 ---
    
    # 提取发电机出力 (Pg)
    # result['gen'] 的每一行对应一个发电机，PG列是出力
    gen_pg = result['gen'][:, PG] 
    
    # 提取总系统负荷
    # result['bus'] 的每一行对应一个节点，PD列是该节点的负荷
    total_load = np.sum(result['bus'][:, PD])
    
    # 也可以提取所有节点的单独负荷，如果想画39条线的话
    # bus_load = result['bus'][:, PD]

    # 返回 发电机出力数组 和 系统总负荷
    return gen_pg, total_load