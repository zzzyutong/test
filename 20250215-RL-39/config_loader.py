# config_loader.py
import json
import os

# 使用全局变量缓存配置，避免多次读取文件
_test_config = None
_train_config = None

def load_config(config_path):
    """
    通用的配置加载函数。
    :param config_path: 配置文件路径
    :return: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'配置文件 {config_path} 不存在')
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f'配置文件 {config_path} 解析错误: {e}')
    return config

def load_test_config(config_path='config/test_config.json'):
    global _test_config
    if _test_config is None:
        _test_config = load_config(config_path)
    return _test_config

def load_train_config(config_path='config/train_config.json'):
    global _train_config
    if _train_config is None:
        _train_config = load_config(config_path)
    return _train_config

# Test Config Getters
def get_test_samples(default=100):
    config = load_test_config()
    return config.get('test_samples', default)

def get_base_load(default=75.2):
    config = load_test_config()
    return config.get('base_load', default)

def get_load_rate(default=1.0):
    config = load_test_config()
    return config.get('load_rate', default)

def get_file_name(default="undefined_file_name"):
    config = load_test_config()
    return config.get('file_name', default)

# Train Config Getters
def get_train_samples(default=100):
    config = load_train_config()
    return config.get('train_samples', default)

def get_train_base_load(default=75.2):
    config = load_train_config()
    return config.get('train_base_load', default)

def get_train_load_rate(default=1.0):
    config = load_train_config()
    return config.get('train_load_rate', default)

def get_train_delayed_load(default=-10.0):
    config = load_train_config()
    return config.get('train_delayed_load', default)

# 你可以根据需要添加更多的获取函数
