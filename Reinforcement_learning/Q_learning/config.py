import sys
from pathlib import Path

curr_path = str(Path().absolute())
parent_path = str(Path().absolute().parent)
sys.path.append(parent_path)  # 添加路径到系统路径

import gym
import torch
import math
import datetime
import numpy as np
from collections import defaultdict
from envs.gridworld_env import CliffWalkingWapper
from agent import QLearning
from common.utils import plot_rewards
from common.utils import save_results, make_dir

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
algo_name = 'Q-learning'  # 算法名称
env_name = 'CliffWalking-v0'  # 环境名称
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU


class QlearningConfig:
    '''训练相关参数'''

    def __init__(self):
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = device  # 检测GPU
        self.train_eps = 400  # 训练的回合数
        self.test_eps = 20  # 测试的回合数
        self.gamma = 0.9  # reward的衰减率
        self.epsilon_start = 0.95  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 300  # e-greedy策略中epsilon的衰减率
        self.lr = 0.1  # 学习率


class PlotConfig:
    ''' 绘图相关参数设置
    '''

    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = device  # 检测GPU
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片
