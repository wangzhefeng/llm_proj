# -*- coding: utf-8 -*-


# ***************************************************
# * File        : A2C.py
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
from keras.utils import to_categorical


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# Define the actor and critic models
state_input = Input(shape=(state_space_size,))
actor = Dense(32, activation='relu')(state_input)
actor = Dense(32, activation='relu')(actor)
actor = Dense(action_space_size, activation='softmax')(actor)
actor_model = Model(inputs=state_input, outputs=actor)
actor_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))

state_input = Input(shape=(state_space_size,))
critic = Dense(32, activation='relu')(state_input)
critic = Dense(32, activation='relu')(critic)
critic = Dense(1, activation='linear')(critic)
critic_model = Model(inputs=state_input, outputs=critic)
critic_model.compile(loss='mse', optimizer=Adam(lr=0.001))


for episode in range(num_episodes):
    current_state = initial_state
    done = False
    while not done:
        # Select an action using the actor model and add exploration noise
        action_probs = actor_model.predict(np.array([current_state]))[0]
        action = np.random.choice(range(action_space_size), p=action_probs)
        
        # Take the action and observe the next state and reward
        next_state, reward, done = take_action(current_state, action)
        
        # Calculate the advantage
        target_value = critic_model.predict(np.array([next_state]))[0][0]
        advantage = reward + gamma * target_value -critic_model.predict(np.array([current_state]))[0][0]
        # Update the actor model
        action_one_hot = to_categorical(action, action_space_size)
        actor_model.train_on_batch(np.array([current_state]), advantage * action_one_hot)
        
        # Update the critic model
        critic_model.train_on_batch(np.array([current_state]), reward + gamma * target_value)
        
        current_state = next_state




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
