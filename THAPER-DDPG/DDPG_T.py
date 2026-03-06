import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import os
import torch
from Actor import Actor
from Critic import Critic
from OUNoise import OrnsteinUhlenbeckActionNoise
import random
from ReplayMemory import ReplayMemory
from priority_memory import Memory


class DDPG_agent:
    def __init__(self, sess, state_shape, action_bound, action_dim, name,
                minibatch_size=128, gamma=0.99, tau=0.001, train_after=200):
        self.actor = Actor(sess, action_bound, action_dim, state_shape, lr=0.0001, tau=tau, name=name)
        self.critic = Critic(sess, state_shape, action_dim, minibatch_size, lr=0.001, tau=tau, name=name)
        self.state_shape = state_shape
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.sess = sess
        self.minibatch_size = minibatch_size
        self.action_bound = action_bound
        self.gamma = gamma
        self.train_after = max(minibatch_size, train_after)
        self.num_action_taken = 0
        self.action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(action_dim))

    def observe(self, state, action, reward, post_state, terminal):
        self.replay_memory.append(state, action, reward, post_state, terminal)

    def act(self, state, info, noise=False):
        action = self.actor.act(state)
        if noise:
            noise = self.action_noise()
            # rand = random.choice((-1, 1))
            # noise = abs(noise) * rand
            if self.num_action_taken < self.train_after:
                noise = noise * 7
                if state[1] > 0:
                    noise[1] = abs(noise[1]) * -2
                    noise[0] = abs(noise[0]) * -2
                else:
                    noise[1] = abs(noise[1]) * 2
                    noise[0] = abs(noise[0]) * -2
                action = np.clip(action + noise, -self.action_bound, self.action_bound)
            else:
                action = np.clip(action + noise, -self.action_bound, self.action_bound)
        else:
            action = np.clip(action, -self.action_bound, self.action_bound)
        if info != None:
            self.num_action_taken += 1
        return action

    def update_target_nets(self):
        # update target net for both actor and critic
        self.sess.run([self.actor.update_ops, self.critic.update_ops])

    def train(self,times = 1):
        if self.num_action_taken >= self.train_after:
            for i in range(times):
                # sample training data from replay memory
                # states, actions, rewards, post_states, terminals = \
                tree_idx, minibatch, ISWeights = self.replay_memory.sample(self.minibatch_size)
                states = [data[0:2] for data in minibatch]
                actions = np.array([data[2:4] for data in minibatch])
                rewards = np.array([data[4] for data in minibatch])
                post_states = [data[5:7] for data in minibatch]
                terminals = np.array([data[7] for data in minibatch])

                # use actor's target network to get action for next state, used to compute target Q value for next state
                mu_post_states = self.actor.target_action(post_states)

                # use critic's target network to get target Q value for next state, then compute actual Q value for current state
                Q_target = self.critic.target_net_eval(post_states, mu_post_states)
                rewards = rewards.reshape([self.minibatch_size, 1])
                terminals = terminals.reshape([self.minibatch_size, 1])
                td_target = rewards + self.gamma * Q_target * (1 - terminals)

                # train critic network
                self.critic.train(states, actions, td_target, ISWeights)

                # update weights in memory
                y = self.critic.current_net_eval(states, actions)
                abs_errors = abs(td_target - y)
                # print("td_target:", td_target, ",y:", y, ",states:", states, ",actions:", actions)
                # print(abs_errors)
                self.replay_memory.batch_update(tree_idx, abs_errors)

                # use actor's current network to get actions for current states, and compute gradient of Q values w.r.t these actions
                pred_actions = self.actor.predict_action(states)
                Q_gradients = self.critic.action_gradient(states, pred_actions) / self.minibatch_size

                # use the computed gradients to train actor network
                self.actor.train(Q_gradients, states)

                # update target networks (critic and actor) with current network parameters
                self.update_target_nets()

    def save(self, saver, dir):
        path = os.path.join(dir, 'model')
        saver.save(self.sess, path)

    def load(self, saver, dir):
        path = os.path.join(dir, 'checkpoint')
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        return False