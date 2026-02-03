import numpy as np
import csv
import os
import pandas as pd
from pypower.api import ppoption, runpf, case39
from config_loader import get_train_base_load
from config_loader import get_train_load_rate

# 定义限制参数
limit_gen = np.array([1040, 646, 725, 652, 508, 687, 580, 564, 865, 1100])
limit_branch = 100.0
limit_voltage = 70

def save_result(result, filename, decimal_places=2):
    formatted_results = {}

    # 格式化结果
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            formatted_results[key] = np.round(value, decimal_places)
        elif isinstance(value, (float, int)):
            formatted_results[key] = round(value, decimal_places)
        else:
            formatted_results[key] = value

    # 定义每类数据的列标题
    column_titles = {
        'bus': ['Bus ID', 'Type', 'Pd', 'Qd', 'Gs', 'Bs', 'Area', 'Vm', 'Va', 'BaseKV', 'Zone', 'Vmax', 'Vmin'],
        'gen': ['Bus ID', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'Status', 'Pmax', 'Pmin', 'Pc1', 'Pc2', 'Qc1min', 'Qc1max', 'Qc2min', 'Qc2max', 'Ramp_agc', 'Ramp_10', 'Ramp_30', 'Ramp_q', 'APF'],
        'branch': ['From Bus', 'To Bus', 'R', 'X', 'B', 'RateA', 'RateB', 'RateC', 'Ratio', 'Angle', 'Status', 'Angmin', 'Angmax', 'Pf', 'Qf', 'Pt', 'Qt'],
        'gencost': ['Model', 'Startup', 'Shutdown', 'N', 'Cost Coefficients'],
    }

    # 写入 CSV 文件
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        for key, value in formatted_results.items():
            writer.writerow([key])  # 写入键作为标题
            if key in column_titles:
                writer.writerow(column_titles[key])  # 写入列标题
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    writer.writerow(value.tolist())  # 写入一维数组
                elif value.ndim == 2:
                    for row in value:
                        writer.writerow(row.tolist())  # 写入二维数组
            else:
                writer.writerow([value])  # 写入单个值

# 计算状态估计的潮流
def state_estimation(load_data, episode, i, flag):
    # 设置最大负荷，gen的限制，branch的限制
    base_load = get_train_base_load()
    load_rate = get_train_load_rate()
    top_load = load_rate * base_load if flag in ['test', 'train_test', 'test1', 'test2', 'test3'] else None
    
    load_data = load_data.cpu()
    filename = f"power_flow/{flag}_{episode}_{i}_se.csv"
    
    if os.path.exists(filename):
        os.remove(filename)
    
    # 变换到0，1
    load_data = (load_data + 1) / 2
    # 变换到0，top_load
    load_data = load_data * top_load + 1
    # 变换成numpy数组
    load_data = load_data.detach().numpy()
    # 截取一位小数，作为负荷load_data_original
    load_data_original = np.round(load_data, decimals=1)
    print(load_data_original)

    data = case39(load_data_original)

    # 设置潮流计算选项
    ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)

    # 执行潮流计算而非最优潮流计算
    result = runpf(data, ppopt)
    save_result(result[0], filename, decimal_places=2)  # runpf返回元组，第一个元素是结果

    # 执行BDD检测
    bdd_result = perform_bdd(filename, flag, episode, i)

    # 获取'gen'和'branch'字段的数据
    gen_data = result[0]['gen']
    branch_data = result[0]['branch']
    voltage_data = result[0]['bus']

    # 统计'gen'字段第2列的gen大于limit_gen的值的个数
    gen_count = np.sum(gen_data[:, 1] > limit_gen)

    # 统计'branch'字段第14列和第16列绝对值大于limit_branch的个数
    branch_count_14 = sum(abs(branch_data[:, 13]) > limit_branch)
    branch_count_16 = sum(abs(branch_data[:, 15]) > limit_branch)
    branch_count = branch_count_14 + branch_count_16

    # 统计'bus'字段电压偏差
    voltage_deviations = abs(voltage_data[:, 7] - 1.0)  # 假设标称电压为1.0 p.u.
    voltage_count = sum(voltage_deviations > 0.05)  # 假设偏差超过5%算作问题

    return gen_count, branch_count, voltage_count, bdd_result

def perform_bdd(filepath, flag, episode, i):
    """
    对39节点系统执行坏数据检测(BDD)
    
    Args:
        filepath: 潮流计算结果文件路径
        flag: 标识符
        episode: 迭代次数
        i: 当前索引
        
    Returns:
        dict: BDD结果
    """
    try:
        # 读取潮流计算结果
        bus_df = pd.read_csv(filepath, skiprows=5, nrows=39, usecols=range(0, 13))
        gen_df = pd.read_csv(filepath, skiprows=46, nrows=10, usecols=range(0, 21))
        branch_df = pd.read_csv(filepath, skiprows=58, nrows=46, usecols=range(0, 17))
        
        # 提取角度值 (Va)
        angle_values = bus_df.iloc[:, 8].to_numpy()
        
        # 提取节点有功功率注入 (Pd)
        p = bus_df.iloc[:, 2].to_numpy()
        
        # 提取发电机有功出力 (Pg)
        g = gen_df.iloc[:, 1].to_numpy()
        
        # 发电机所在节点编号
        g_n = gen_df.iloc[:, 0].to_numpy()
        
        # 计算观测值z1 (节点功率注入)
        z1 = -p.copy()  # 负载取负值
        for i, g_n_value in enumerate(g_n):
            if 1 <= g_n_value <= 39:
                z1[int(g_n_value) - 1] += g[i]  # 加上发电机出力
                
        # 从case39获取网络参数
        case_data = case39()
        branch_data = case_data['branch']
        
        # 提取线路参数
        num_buses = 39
        num_lines = len(branch_data)
        
        # 初始化电导矩阵B
        B = np.zeros((num_buses, num_buses))
        
        # 将角度从度转换为弧度
        x1 = np.radians(angle_values)
        
        # 构建电导矩阵B
        for line_idx in range(num_lines):
            line = branch_data[line_idx]
            i, j = int(line[0]) - 1, int(line[1]) - 1  # 节点编号从1开始，转为从0开始
            r, x = line[2], line[3]  # 电阻和电抗
            
            if x != 0:  # 避免除零错误
                angle_diff = x1[i] - x1[j]  # 计算角度差
                B_ij = 1 / x * np.cos(angle_diff)  # 电抗的倒数乘以角度差的余弦值
                B[i, j] = -B_ij  # 非对角线元素是负的电导
                B[j, i] = -B_ij
                B[i, i] += B_ij  # 对角线元素是连接到节点的所有电导之和
                B[j, j] += B_ij
        
        # 计算基于状态的功率注入估计值z2
        z2 = np.dot(B, x1)
        # 转换单位 (100MW, 2/根号三)
        z2 *= 100
        z2 *= 1.1547
        
        # 计算残差平方和
        r = np.sum(((z1 - z2) ** 2))
        
        # 保存BDD结果到CSV文件
        bdd_filename = f"bdd_results/{flag}_{episode}_{i}_bdd.csv"
        os.makedirs("bdd_results", exist_ok=True)
        
        with open(bdd_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Bus", "z1 (Observed Power)", "z2 (Estimated Power)", "Residual", "Angle (deg)"])
            for bus_idx in range(num_buses):
                writer.writerow([
                    bus_idx + 1,  # 节点编号
                    z1[bus_idx],  # 观测功率
                    z2[bus_idx],  # 估计功率
                    z1[bus_idx] - z2[bus_idx],  # 残差
                    angle_values[bus_idx]  # 角度(度)
                ])
            
            # 添加总体统计信息
            writer.writerow([])
            writer.writerow(["Total Residual Square Sum", r])
        
        # 返回BDD结果
        return {
            "residual_sum": r,
            "z1": z1,
            "z2": z2,
            "angles": angle_values
        }
        
    except Exception as e:
        print(f"Error in BDD for {filepath}: {e}")
        return {
            "residual_sum": -1,  # 错误标志
            "error": str(e)
        }

# 如果直接运行此脚本
if __name__ == "__main__":
    # 这里可以添加测试代码
    pass
