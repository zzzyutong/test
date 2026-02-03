import torch
import torch.nn.functional as F
import train_power_flow
import se
import numpy as np
import os

class PowerSystemEnv:
    def __init__(self, device, max_episodes=1000, max_steps=750):
        self.device = device
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.current_step = 0
        self.data_loader = self.load_data()
        self.data_iter = iter(self.data_loader)
        self.fake_data_history = None  # 用来存储历史 fake_data
        self.reset()

    def load_data(self):
        import data_load
        return data_load.load_data_from_csv()


    # 状态 = 观测
    def reset(self):
        try:
            real_data, = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            real_data, = next(self.data_iter)
        self.real_data = real_data.to(self.device)
        self.current_step = 0
        self.fake_data_history = self.real_data.clone()  # 初始化 fake_data 存储
        # 返回初始状态
        return self.real_data

    def step(self, action, episode, step_idx):
        """
        action: 生成的扰动量
        """

        print(self.real_data)

        # 生成虚假数据
        fake_data = self.real_data + action

        print(fake_data)

        # 确保数据在合理范围内并进行检查
        fake_data = self._process_fake_data(fake_data)

        #DoS检测



        # 计算功率潮流
        gen_count, branch_count, voltage_count = train_power_flow.loadflow(fake_data[0], episode, step_idx, "train_test")

        # 计算奖励
        reward = self.calculate_reward(gen_count, branch_count, voltage_count, action)

        # 获取下一个状态
        try:
            next_real_data, = next(self.data_iter)
            next_state = next_real_data.to(self.device)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            next_real_data, = next(self.data_iter)
            next_state = next_real_data.to(self.device)

        # 判断是否结束
        done = False
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        self.real_data = next_state  # 更新当前数据样本

        return next_state, reward, done

    def _process_fake_data(self, fake_data):
        """
        处理 fake_data: 检查是否有小于 -5 的值，并确保数据在合理范围内
        """
        # 检查并替换小于 -5 的值
        fake_data = self._replace_invalid_data(fake_data)

        print(fake_data)

        # 限制数据范围
        fake_data = torch.clamp(fake_data, -1, 1)

        print(fake_data)

        # 更新 fake_data 存储
        self.fake_data_history = fake_data.clone()

        return fake_data

    def _replace_invalid_data(self, fake_data):
        """
        替换 fake_data 中小于 -5 的元素为存储中的对应元素
        """
        invalid_mask = fake_data < -5
        fake_data[invalid_mask] = self.fake_data_history[invalid_mask]
        return fake_data

    def calculate_reward(self, gen_count, branch_count, voltage_count, action):
        """
        根据攻击效果和隐蔽性计算奖励
        """
        # 权重可以根据具体需求调整
        w_gen = 0.3
        w_branch = 0.3
        w_voltage = 0
        w_l1 = 0.1
        w_l2 = 0.3

        # 攻击效果
        attack_effect = w_gen * gen_count + w_branch * branch_count + w_voltage * voltage_count

        # 隐蔽性（L1正则化）
        l1_regularization = torch.mean(torch.abs(action))

        # 计算动作的 L2 范数
        action_l2_norm = torch.norm(action, p=2).item()

        # 定义 L2 范数的奖励范围
        l2_min = 0.3 * 0.03  # 0.015
        l2_max = 0.3 * 0.12  # 0.045

        # 根据 L2 范数确定正负奖励
        if l2_min <= action_l2_norm <= l2_max:
            l2_reward = 1.0  # 正奖励
        else:
            l2_reward = -1.0  # 负奖励

        # 综合奖励
        # 奖励：攻击效果越大越好，扰动越小越好
        reward = attack_effect - w_l1 * l1_regularization.item() + w_l2 * l2_reward
        print(f"attack_effect:{attack_effect}")
        print(f"l1_regularization:{l1_regularization}")
        print(f"l2_reward:{l2_reward}")
        print(f"reward:{reward}")
        return reward
