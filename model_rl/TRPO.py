# -*- coding: utf-8 -*-


# ***************************************************
# * File        : TRPO.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-12
# * Version     : 0.1.041215
# * Description : description
# * Link        : https://mp.weixin.qq.com/s/UrHZfG5tOzV9PmHLhMHdRQ
# * Requirement : pip install baselines
# ***************************************************


# python libraries
import os
import sys

import gym
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.trpo_mpi import trpo_mpi
import tensorflow as tf
import gym


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# Define the policy network
class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


# Initialize the environment
env = gym.make("CartPole-v1")
env = DummyVecEnv([lambda: env])
# Define the policy network
policy_fn = mlp_policy
# Train the TRPO model
model = trpo_mpi.learn(env, policy_fn, max_iters = 1000)


# Initialize the environment
env = gym.make("CartPole-v1")
# Initialize the policy network
policy_network = PolicyNetwork()
# Define the optimizer
optimizer = tf.optimizers.Adam()
# Define the loss function
loss_fn = tf.losses.BinaryCrossentropy()


max_iters = 1000  # maximum number of iterations
for i in range(max_iters):
    # Sample an action from the policy network
    action = tf.squeeze(tf.random.categorical(policy_network(observation), 1))
    # Take a step in the environment
    observation, reward, done, _ = env.step(action)
    with tf.GradientTape() as tape:
            # Compute the loss
            loss = loss_fn(reward, policy_network(observation))
            # Compute the gradients
            grads = tape.gradient(loss, policy_network.trainable_variables)
            # Perform the update step
            optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))
            if done:
                # Reset the environment
                observation = env.reset()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
