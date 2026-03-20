import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import numpy as np

class Actor:
    def __init__(self, sess, action_bound, action_dim, state_shape, name, lr=1e-4, tau=0.001, prior_loss_coef=0.05):
        self.sess = sess
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_shape = state_shape
        self.tau = tau
        self.prior_loss_coef = prior_loss_coef

        self.state = tf.placeholder(tf.float32, [None, state_shape])
        self.img = tf.placeholder(tf.float32, [None, 64, 64, 1])
        self.post_state = tf.placeholder(tf.float32, [None, state_shape])
        self.post_img = tf.placeholder(tf.float32, [None, 64, 64, 1])
        self.Q_gradient = tf.placeholder(tf.float32, [None, action_dim])
        self.prior_action0 = tf.placeholder(tf.float32, [None, 1])
        self.prior_weight = tf.placeholder(tf.float32, [None, 1])

        with tf.variable_scope(name + "actor"):
            self.eval_net = self._build_network(self.state, "eval_net")
            self.target_net = self._build_network(self.post_state, "target_net")

        self.eval_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + "actor/eval_net")
        self.target_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + "actor/target_net")

        self.policy_gradient = tf.gradients(ys=self.eval_net, xs=self.eval_param, grad_ys=-self.Q_gradient)
        self.forward_action = self.eval_net[:, 0:1]
        self.prior_loss = tf.reduce_mean(self.prior_weight * tf.square(self.forward_action - self.prior_action0))
        self.prior_gradients = tf.gradients(self.prior_loss_coef * self.prior_loss, self.eval_param)

        self.total_gradients = []
        for pg, rg, param in zip(self.policy_gradient, self.prior_gradients, self.eval_param):
            if pg is None and rg is None:
                grad = tf.zeros_like(param)
            elif pg is None:
                grad = rg
            elif rg is None:
                grad = pg
            else:
                grad = pg + rg
            self.total_gradients.append(grad)

        self.train_step = tf.train.AdamOptimizer(lr).apply_gradients(zip(self.total_gradients, self.eval_param))
        self.update_ops = self._update_target_net_op()

    def _build_network(self, X, scope):
        with tf.variable_scope(scope):
            init_w1 = tf.random_uniform_initializer(-0.05, 0.05)
            concat = tf.concat([X], 1)
            fc1 = tf.layers.dense(inputs=concat, units=200, activation=tf.nn.relu, kernel_initializer=init_w1)
            fc2 = tf.layers.dense(inputs=fc1, units=200, activation=tf.nn.relu, kernel_initializer=init_w1)
            fc3 = tf.layers.dense(inputs=fc2, units=200, activation=tf.nn.relu, kernel_initializer=init_w1)
            action_normal = tf.layers.dense(inputs=fc3, units=self.action_dim, activation=tf.nn.tanh, kernel_initializer=init_w1)
            action = tf.multiply(action_normal, self.action_bound)
        return action

    def act(self, state):
        state = np.reshape(state, [1, self.state_shape])
        action = self.sess.run(self.eval_net, feed_dict={self.state: state})[0]
        return action

    def predict_action(self, states):
        return self.sess.run(self.eval_net, feed_dict={self.state: states})

    def target_action(self, post_states):
        return self.sess.run(self.target_net, feed_dict={self.post_state: post_states})

    def train(self, Q_gradient, states, prior_action0=None, prior_weight=None):
        if prior_action0 is None:
            prior_action0 = np.zeros((states.shape[0], 1), dtype=np.float32)
        if prior_weight is None:
            prior_weight = np.zeros((states.shape[0], 1), dtype=np.float32)
        self.sess.run(
            self.train_step,
            feed_dict={
                self.state: states,
                self.Q_gradient: Q_gradient,
                self.prior_action0: prior_action0,
                self.prior_weight: prior_weight,
            }
        )

    def get_prior_loss(self, states, prior_action0, prior_weight):
        return self.sess.run(
            self.prior_loss,
            feed_dict={
                self.state: states,
                self.prior_action0: prior_action0,
                self.prior_weight: prior_weight,
            }
        )

    def _update_target_net_op(self):
        return [
            tf.assign(dest_var, (1 - self.tau) * dest_var + self.tau * src_var)
            for dest_var, src_var in zip(self.target_param, self.eval_param)
        ]
