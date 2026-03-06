import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

class Critic:
    def __init__(self, sess, state_shape, action_dim, minibatch_size, name, lr=1e-3, tau=0.001):
        self.sess = sess
        self.tau = tau
        self.minibatch_size = minibatch_size
        
        self.reward = tf.placeholder(tf.float32, [None, 1])  # reward
        self.td_target = tf.placeholder(tf.float32, [None, 1])  # actual Q value
        
        # inputs for current network
        self.state = tf.placeholder(tf.float32, [None, state_shape])  # state input (excluding image)
        self.img = tf.placeholder(tf.float32, [None, 64, 64, 1])  # image input
        self.action = tf.placeholder(tf.float32, [None, action_dim])  # action input
        
        # inputs for target network
        self.t_state = tf.placeholder(tf.float32, [None, state_shape])  # state input
        self.t_img = tf.placeholder(tf.float32, [None, 64, 64, 1])  # image input
        self.t_action = tf.placeholder(tf.float32, [None, action_dim])  # action input
        self.is_weight = tf.placeholder(tf.float32, [None, 1])
        
        with tf.variable_scope(name + "critic"):
            # self.eval_net = self._build_network(self.state, self.action, self.img, "eval_net")  # create current network
            self.eval_net = self._build_network(self.state, self.action, "eval_net")
            # self.target_net = self._build_network(self.t_state, self.t_action, self.t_img, "target_net")  # create target network
            self.target_net = self._build_network(self.t_state, self.t_action, "target_net")  # create target network
        
        self.eval_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + "critic/eval_net")  # get current network variables
        self.target_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + "critic/target_net")  # get target network variables
        
        self.loss = tf.losses.mean_squared_error(self.td_target, self.eval_net)  # mean squared error loss
        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss * self.is_weight)
        self.action_gradients = tf.gradients(self.eval_net, self.action)  # gradient of Q value w.r.t action
        
        self.update_ops = self._update_target_net_op()  # assign current network parameters to target network
        
    # def _build_network(self, X, action, image, scope):
        # with tf.variable_scope(scope):
        #     init_w1 = tf.truncated_normal_initializer(0., 3e-4)  # generate random numbers from normal distribution (but within two standard deviations, others truncated)
        #     init_w2 = tf.random_uniform_initializer(-0.05, 0.05)  # also generate random numbers from uniform distribution
        #
        #     conv1 = tf.layers.conv2d(image, 32, [5,5], strides=[2,2], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)  # convolution 32*32
        #     pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)  # pooling 16*16
        #     conv2 = tf.layers.conv2d(pool1, 32, [5,5], strides=[1,1], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)  # 16*16
        #     pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)  # 8*8
        #     conv3 = tf.layers.conv2d(pool2, 32, [5,5], strides=[1,1], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)  # 8*8
        #     pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)  # 4*4
        #     flatten = tf.layers.flatten(pool3) # shape(None, 4*4*32)
        #     concat = tf.concat([flatten, action, X], 1)
        #
        #     # create fully connected neural network
        #     fc1 = tf.layers.dense(inputs=concat, units=200, activation=tf.nn.relu, kernel_initializer=init_w2)
        #     fc2 = tf.layers.dense(inputs=fc1, units=200, activation=tf.nn.relu, kernel_initializer=init_w2)
        #     fc3 = tf.layers.dense(inputs=fc2, units=200, activation=tf.nn.relu, kernel_initializer=init_w2)
        #     Q = tf.layers.dense(inputs=fc3, units=1, kernel_initializer=init_w2)
        # return Q
    def _build_network(self, X, action, scope):
        with tf.variable_scope(scope):
            init_w1 = tf.random_uniform_initializer(-0.05, 0.05)  # also generate random numbers from uniform distribution

            concat = tf.concat([action, X], 1)

            # create fully connected neural network
            fc1 = tf.layers.dense(inputs=concat, units=200, activation=tf.nn.relu, kernel_initializer=init_w1)
            fc2 = tf.layers.dense(inputs=fc1, units=200, activation=tf.nn.relu, kernel_initializer=init_w1)
            fc3 = tf.layers.dense(inputs=fc2, units=200, activation=tf.nn.relu, kernel_initializer=init_w1)
            Q = tf.layers.dense(inputs=fc3, units=1, kernel_initializer=init_w1)
        return Q
    # compute Q value for given state-action pair using target network
    def target_net_eval(self, states, actions):
        # imgs, dstates = self._seperate_image(states)
        Q_target = self.sess.run(self.target_net, feed_dict={self.t_state:states, self.t_action:actions})
        return Q_target
    # compute Q value for given state-action pair using current network
    def current_net_eval(self, states, actions):
        # imgs, dstates = self._seperate_image(states)
        Q_target = self.sess.run(self.eval_net, feed_dict={self.state:states, self.action:actions})
        return Q_target
    # gradient of Q value w.r.t action
    def action_gradient(self, states, actions):
        # imgs, dstates = self._seperate_image(states)
        return self.sess.run(self.action_gradients, feed_dict={self.state:states, self.action:actions})[0]
    # optimize current network
    def train(self, states, actions, td_target, is_weight):
        # imgs, dstates = self._seperate_image(states)
        actions = actions.reshape([self.minibatch_size,2])
        feed_dict = {self.state:states, self.action:actions, self.td_target:td_target, self.is_weight:is_weight}
        self.sess.run(self.train_step, feed_dict=feed_dict)
    # assign current network parameters to target network
    def _update_target_net_op(self):
        ops = [tf.assign(dest_var, (1-self.tau) * dest_var + self.tau * src_var)
               for dest_var, src_var in zip(self.target_param, self.eval_param)]
        return ops
    # separate image data from state
    def _seperate_image(self, states):
        images = np.array([state[0] for state in states])
        dstates = np.array([state[1] for state in states])
        return images, dstates