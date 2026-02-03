import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import pandas as pd

# 注意：在这个代码中，你需要提供你自己的真实数据集路径 `file_path`。
# 加载数据
file_path = 'dataset/data_to_train.csv'  # 指定CSV文件的路径

# 检查是否有可用的CUDA设备
device = torch.device("cpu")
print(f"Loading on device: {device}")

# 设置读取频率hours，对应hours的行数
hours = 1
lines = 12 * hours

# 读取CSV文件
def load_data_from_csv():
    # 使用pandas读取csv文件
    df = pd.read_csv(file_path, header=None, skiprows=lambda x: x % lines != 0)  # 每隔12行读取一次，跳过不需要的行
    # 将数据转换为numpy数组
    data = df.values.astype('float32')
    # 将numpy数组转换为torch张量
    tensor_data = torch.from_numpy(data)
    # 将数据移动到设备
    real_data_samples = tensor_data.to(device)
    # 创建DataLoader，每个批次只有1行数据
    dataloader = DataLoader(TensorDataset(real_data_samples), batch_size=1, shuffle=False)

    return dataloader
