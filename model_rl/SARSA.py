# -*- coding: utf-8 -*-


# ***************************************************
# * File        : SARSA.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-12
# * Version     : 0.1.041215
# * Description : description
# * Link        : https://mp.weixin.qq.com/s/UrHZfG5tOzV9PmHLhMHdRQ
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# Define the Q-table and the learning rate
Q = np.zeros((state_space_size, action_space_size))
alpha = 0.1

# Define the exploration rate and discount factor
epsilon = 0.1
gamma = 0.99

for episode in range(num_episodes):
    current_state = initial_state
    action = epsilon_greedy_policy(epsilon, Q, current_state)
    while not done:
        # Take the action and observe the next state and reward
        pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
