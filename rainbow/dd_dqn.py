import tensorflow as tf
import numpy as np

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
from collections import deque

import random
import gym
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
     if 'bhs' in env:
          print('Remove {} from registry'.format(env))
          del gym.envs.registration.registry.env_specs[env]
import bhs

import matplotlib.pyplot as plt

class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.layer2 = tf.keras.layers.Dense(64, activation='relu')
        self.state = tf.keras.layers.Dense(1)
        self.action = tf.keras.layers.Dense(128)

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        state = self.state(layer2)
        action = self.action(layer2)
        mean = tf.keras.backend.mean(action, keepdims=True)
        advantage = (action - mean)
        value = state + advantage
        return value

class Agent:
    def __init__(self):
        # hyper parameters
        self.lr =0.001
        self.gamma = 0.99

        self.dqn_model = DQN()
        self.dqn_target = DQN()
        self.opt = optimizers.Adam(lr=self.lr, )

        self.batch_size = 32
        self.state_size = [101,7]
        self.action_size = 128

        self.memory = deque(maxlen=2000)

    def update_target(self):
        self.dqn_target.set_weights(self.dqn_model.get_weights())

    def get_action(self, state, epsilon):
        q_value = self.dqn_model(tf.convert_to_tensor([state], dtype=tf.float32))[0]
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(q_value) 
        return action, q_value

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states = [i[0] for i in mini_batch]
        actions = [i[1] for i in mini_batch]
        rewards = [i[2] for i in mini_batch]
        next_states = [i[3] for i in mini_batch]
        dones = [i[4] for i in mini_batch]

        dqn_variable = self.dqn_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)

            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            target_q = self.dqn_target(tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32))
            main_q = self.dqn_model(tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32))
            main_q = tf.stop_gradient(main_q)
            next_action = tf.argmax(main_q, axis=1)
            target_value = tf.reduce_sum(tf.one_hot(next_action, self.action_size) * target_q, axis=1)

            target_value = (1-dones) * self.gamma * target_value + rewards

            main_q = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            main_value = tf.reduce_sum(tf.one_hot(actions, self.action_size) * main_q, axis=1)

            error = tf.square(main_value - target_value) * 0.5
            error = tf.reduce_mean(error)
            
        dqn_grads = tape.gradient(error, dqn_variable)
        self.opt.apply_gradients(zip(dqn_grads, dqn_variable))


    def run(self):
        test_frq = 2000
        env = gym.make('bhs-v0')
        episode = 0
        step = 0
        env.reset(total=True)
        test_scores = []
        while True:
            state = env.reset(total=env.deadlock)
            done = False
            episode += 1
            epsilon = 1 / (episode * 0.01 + 1)
            score = 0
            while not done:
                step += 1
                action, q_value = self.get_action(state, epsilon)
                next_state, reward, done, info = env.step(action)

                self.append_sample(state, action, reward, next_state, env.deadlock)
                
                score += reward

                state = next_state

                if step > 100:
                    self.update()
                    if step % 200 == 0:
                        self.update_target()
            
                if step % test_frq == 0:
                    test_score=self.test(seed=1000+len(test_scores))
                    test_scores.append(test_score)
                    x=np.arange(test_frq, step+1, test_frq)
                    plt.plot(x,test_scores)
                    plt.legend(['10','30','50','70','90'])
                    plt.title("Score at step: " +str(step))
                    plt.show()
                    plt.pause(0.05)    
            
            print(episode, "{:.3f}".format(epsilon), score)

    def test(self, seed=1000):
        print("Testing")
        env = gym.make('bhs-v0')
        scores = []
        for i in [10,30,50,70,90]:
            obs = env.reset(total=True, seed=seed, numtotes = i)
            score = 0
            done=False
            step=0
            while not done:
                step+=1
                # Choose action
                action, _ = self.get_action(obs,0)
                # print(obs,action)
                # print(action)
                # Perform action in environment and take a step
                obs, rew, done, _ = env.step(action)
                score += rew
                
            print("Number of steps: ", step)
            print("Score: ", score)
            scores.append(score)
            # print("RL",score)
            # score=0
            # _=dqn.env.reset(total=True, seed=seed, numtotes = i)
            # for step in range(args.steplimit):
            #     # Choose action
            #     # action = dqn.choose_action(obs)
            #     # print(action)
            #     # Perform action in environment and take a step
            #     obs, rew, done, _, _ = dqn.env.step(shortestPath=True)
            #     score += rew
            #     if done:
            #         break
            # print("SSP",score)
            # score=0
            # _=dqn.env.reset(total=True, seed=seed, numtotes = i)
            # for step in range(args.steplimit):
            #     # Choose action
            #     # action = dqn.choose_action(obs)
            #     # print(action)
            #     # Perform action in environment and take a step
            #     obs, rew, done, _, _ = dqn.env.step(shortestPath=True, dynamic=True)
            #     score += rew
            #     if done:
            #         break
            # print("DSP",score)
            # score=0
            # _=dqn.env.reset(total=True, seed=seed, numtotes = i)
            # for step in range(args.steplimit):
            #     # Choose action
            #     # action = dqn.choose_action(obs)
            #     # print(action)
            #     # Perform action in environment and take a step
            #     obs, rew, done, _, _ = dqn.env.step(shortestPath=True, dynamic=True, dla=True)
            #     score += rew
            #     if done:
            #         break
            # print("DSPdla",score)
            
        print("Done testing")
        return scores

if __name__ == '__main__':
    agent = Agent()
    agent.run()
