import os
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from .actor import Actor
from .critic import Critic
from .modules.exploration_scheduler import get_noise_scale
from .replay.per_memory import Memory
from .modules.state_adapter import COORD_FEATURE_SCALE

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.25, theta=.5, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class TunnelAwareDDPGAgent:
    replay_memory = None

    def __init__(self, sess, state_shape, action_bound, action_dim, name,
                 minibatch_size=128, gamma=0.99, tau=0.001, train_after=200,
                 prior_loss_coef=0.05, k_form=0.5, delta_s=1.5, safe_dist_m=4.5):
        self.actor = Actor(sess, action_bound, action_dim, state_shape, lr=0.0001, tau=tau, name=name,
                           prior_loss_coef=prior_loss_coef)
        self.critic = Critic(sess, state_shape, action_dim, minibatch_size, lr=0.001, tau=tau, name=name)
        self.state_shape = state_shape
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.sess = sess
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.train_after = max(minibatch_size, train_after)
        self.num_action_taken = 0
        self.action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(action_dim))
        self.prior_loss_coef = prior_loss_coef
        self.k_form = k_form
        self.delta_s = delta_s
        self.safe_dist_m = safe_dist_m
        self.coord_feature_scale = float(COORD_FEATURE_SCALE)

    @classmethod
    def initialize_replay_memory(cls, capacity, transition_len):
        cls.replay_memory = Memory(capacity, transition_len)

    def act(self, state, info, noise=False, consecutive_success=0):
        action = self.actor.act(state)
        if noise:
            noise_sample = self.action_noise()
            noise_scale = get_noise_scale(consecutive_success)
            noise_sample = noise_sample * noise_scale
            if self.num_action_taken < self.train_after and noise_scale > 0:
                noise_sample = noise_sample * 7
                if state[1] > 0:
                    noise_sample[1] = abs(noise_sample[1]) * -2
                    noise_sample[0] = abs(noise_sample[0]) * -2
                else:
                    noise_sample[1] = abs(noise_sample[1]) * 2
                    noise_sample[0] = abs(noise_sample[0]) * -2
                action = np.clip(noise_sample, -self.action_bound, self.action_bound)
            else:
                action = np.clip(action + noise_sample, -self.action_bound, self.action_bound)
        else:
            action = np.clip(action, -self.action_bound, self.action_bound)
        if info is not None:
            self.num_action_taken += 1
        return action

    def update_target_nets(self):
        self.sess.run([self.actor.update_ops, self.critic.update_ops])

    def _build_risk_aware_prior_targets(self, states, pred_actions):
        states = np.asarray(states, dtype=np.float32)
        pred_actions = np.asarray(pred_actions, dtype=np.float32)

        d_obs_signed_norm = states[:, 0]
        ds1 = states[:, 4]
        ds2 = states[:, 5]

        err = -(ds1 + ds2) * self.coord_feature_scale / 3.0
        no_obstacle = (d_obs_signed_norm == -1)
        d_obs_m = np.abs(d_obs_signed_norm) * 4.0
        allow_guidance = np.logical_or(no_obstacle, d_obs_m > self.safe_dist_m)
        need_coord = np.abs(err) > self.delta_s
        use_prior = np.logical_and(allow_guidance, need_coord)

        prior_action0 = pred_actions[:, 0:1].copy()
        delta_a = (-self.k_form * err).reshape(-1, 1)
        prior_action0 = np.where(use_prior.reshape(-1, 1), prior_action0 + delta_a, prior_action0)
        prior_action0 = np.clip(prior_action0, -self.action_bound, self.action_bound).astype(np.float32)

        prior_weight = use_prior.astype(np.float32).reshape(-1, 1)
        return prior_action0, prior_weight

    def train(self, times=1):
        if self.num_action_taken >= self.train_after:
            for _ in range(times):
                tree_idx, minibatch, ISWeights = self.replay_memory.sample(self.minibatch_size)
                sdim = self.state_shape
                adim = self.action_dim
                states = np.array([data[0:sdim] for data in minibatch], dtype=np.float32)
                actions = np.array([data[sdim:sdim + adim] for data in minibatch], dtype=np.float32)
                rewards = np.array([data[sdim + adim] for data in minibatch], dtype=np.float32)
                post_states = np.array([data[sdim + adim + 1:sdim + adim + 1 + sdim] for data in minibatch], dtype=np.float32)
                terminals = np.array([data[sdim + adim + 1 + sdim] for data in minibatch], dtype=np.float32)
                mu_post_states = self.actor.target_action(post_states)
                Q_target = self.critic.target_net_eval(post_states, mu_post_states)
                rewards = rewards.reshape([self.minibatch_size, 1])
                terminals = terminals.reshape([self.minibatch_size, 1])
                td_target = rewards + self.gamma * Q_target * (1 - terminals)
                self.critic.train(states, actions, td_target, ISWeights)
                y = self.critic.current_net_eval(states, actions)
                abs_errors = abs(td_target - y)
                self.replay_memory.batch_update(tree_idx, abs_errors)
                pred_actions = self.actor.predict_action(states)
                prior_action0, prior_weight = self._build_risk_aware_prior_targets(states, pred_actions)
                Q_gradients = self.critic.action_gradient(states, pred_actions) / self.minibatch_size
                self.actor.train(Q_gradients, states, prior_action0=prior_action0, prior_weight=prior_weight)
                self.update_target_nets()

    def save(self, saver, dir_path):
        path = os.path.join(dir_path, 'model')
        saver.save(self.sess, path)

    def load(self, saver, dir_path):
        path = os.path.join(dir_path, 'checkpoint')
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        return False

DDPG_agent = TunnelAwareDDPGAgent
