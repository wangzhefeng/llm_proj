# -*- coding: utf-8 -*-


# ***************************************************
# * File        : PRO.py
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
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# Define the policy model
state_input = Input(shape=(state_space_size,))
policy = Dense(32, activation='relu')(state_input)
policy = Dense(32, activation='relu')(policy)
policy = Dense(action_space_size, activation='softmax')(policy)
policy_model = Model(inputs=state_input, outputs=policy)

# Define the value model
value_model = Model(inputs=state_input, outputs=Dense(1, activation='linear')(policy))

# Define the optimizer
optimizer = Adam(lr=0.001)

for episode in range(num_episodes):
    current_state = initial_state
    while not done:
        # Select an action using the policy model
        action_probs = policy_model.predict(np.array([current_state]))[0]
        action = np.random.choice(range(action_space_size), p=action_probs)
        # Take the action and observe the next state and reward
        next_state, reward, done = take_action(current_state, action)
        
        # Calculate the advantage
        target_value = value_model.predict(np.array([next_state]))[0][0]
        advantage = reward + gamma * target_value -value_model.predict(np.array([current_state]))[0][0]
        
        # Calculate the old and new policy probabilities
        old_policy_prob = action_probs[action]
        new_policy_prob = policy_model.predict(np.array([next_state]))[0][action]
        
        # Calculate the ratio and the surrogate loss
        ratio = new_policy_prob / old_policy_prob
        surrogate_loss = np.minimum(ratio * advantage, np.clip(ratio, 1 - epsilon, 1 + epsilon) *advantage)

        # Update the policy and value models
        policy_model.trainable_weights = value_model.trainable_weights
        policy_model.compile(optimizer=optimizer, loss=-surrogate_loss)
        policy_model.train_on_batch(np.array([current_state]), np.array([action_one_hot]))
        value_model.train_on_batch(np.array([current_state]), reward + gamma * target_value)

        current_state = next_state




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
