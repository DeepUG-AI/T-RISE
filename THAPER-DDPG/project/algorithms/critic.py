import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import numpy as np

class Critic:
    def __init__(self, sess, state_shape, action_dim, minibatch_size, name, lr=1e-3, tau=0.001):
        self.sess = sess
        self.tau = tau
        self.minibatch_size = minibatch_size

        self.reward = tf.placeholder(tf.float32, [None, 1])
        self.td_target = tf.placeholder(tf.float32, [None, 1])
        self.state = tf.placeholder(tf.float32, [None, state_shape])
        self.img = tf.placeholder(tf.float32, [None, 64, 64, 1])
        self.action = tf.placeholder(tf.float32, [None, action_dim])
        self.t_state = tf.placeholder(tf.float32, [None, state_shape])
        self.t_img = tf.placeholder(tf.float32, [None, 64, 64, 1])
        self.t_action = tf.placeholder(tf.float32, [None, action_dim])
        self.is_weight = tf.placeholder(tf.float32, [None, 1])

        with tf.variable_scope(name + "critic"):
            self.eval_net = self._build_network(self.state, self.action, "eval_net")
            self.target_net = self._build_network(self.t_state, self.t_action, "target_net")

        self.eval_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + "critic/eval_net")
        self.target_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + "critic/target_net")

        self.loss = tf.losses.mean_squared_error(self.td_target, self.eval_net)
        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss * self.is_weight)
        self.action_gradients = tf.gradients(self.eval_net, self.action)
        self.update_ops = self._update_target_net_op()

    def _build_network(self, X, action, scope):
        with tf.variable_scope(scope):
            init_w1 = tf.random_uniform_initializer(-0.05, 0.05)
            concat = tf.concat([action, X], 1)
            fc1 = tf.layers.dense(inputs=concat, units=200, activation=tf.nn.relu, kernel_initializer=init_w1)
            fc2 = tf.layers.dense(inputs=fc1, units=200, activation=tf.nn.relu, kernel_initializer=init_w1)
            fc3 = tf.layers.dense(inputs=fc2, units=200, activation=tf.nn.relu, kernel_initializer=init_w1)
            Q = tf.layers.dense(inputs=fc3, units=1, kernel_initializer=init_w1)
        return Q

    def target_net_eval(self, states, actions):
        return self.sess.run(self.target_net, feed_dict={self.t_state: states, self.t_action: actions})

    def current_net_eval(self, states, actions):
        return self.sess.run(self.eval_net, feed_dict={self.state: states, self.action: actions})

    def action_gradient(self, states, actions):
        return self.sess.run(self.action_gradients, feed_dict={self.state: states, self.action: actions})[0]

    def train(self, states, actions, td_target, is_weight):
        actions = actions.reshape([self.minibatch_size, 2])
        feed_dict = {self.state: states, self.action: actions, self.td_target: td_target, self.is_weight: is_weight}
        self.sess.run(self.train_step, feed_dict=feed_dict)

    def _update_target_net_op(self):
        return [
            tf.assign(dest_var, (1 - self.tau) * dest_var + self.tau * src_var)
            for dest_var, src_var in zip(self.target_param, self.eval_param)
        ]
