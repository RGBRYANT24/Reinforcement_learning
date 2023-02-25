import sys
from pathlib import Path
curr_path = str(Path().absolute())
parent_path = str(Path().absolute().parent)
sys.path.append(parent_path) # 添加路径到系统路径

import gym
import torch
import math
import datetime
import numpy as np
import config
from collections import defaultdict
from envs.gridworld_env import CliffWalkingWapper
from agent import QLearning
from common.utils import plot_rewards
from common.utils import save_results,make_dir

def train(cfg, env, agent):
    rewards = [] #记录每次训练的奖励
    ma_rewards = [] # 记录滑动平滑奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0 #记录每个回合的奖励
        state = env.reset()
        while True: #直到游戏结束
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 20 == 0:
            print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
    print('完成训练！')
    return rewards, ma_rewards


def env_agent_config(cfg, seed=1): # 创建智能体
    env = gym.make(cfg.env_name)
    env = CliffWalkingWapper(env)
    env.seed(seed)  # 设置随机种子
    state_dim = env.observation_space.n  # 状态维度
    action_dim = env.action_space.n  # 动作维度
    agent = QLearning(state_dim, action_dim, cfg)
    return env, agent

if __name__ == '__main__':
    cfg = config.QlearningConfig()
    plot_cfg = config.PlotConfig()
    env, agent = env_agent_config(cfg, seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)  # 创建保存结果和模型路径的文件夹
    agent.save(path=plot_cfg.model_path)  # 保存模型
    save_results(rewards, ma_rewards, tag='train',
                 path=plot_cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果


