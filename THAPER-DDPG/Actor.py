import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

class Actor:
    def __init__(self, sess, action_bound, action_dim, state_shape, name, lr=1e-4, tau=0.001):
        self.sess = sess
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_shape = state_shape
        self.tau = tau
        
        self.state = tf.placeholder(tf.float32, [None, state_shape])  # state values (excluding image)
        self.img = tf.placeholder(tf.float32, [None, 64, 64, 1])  # image
        self.post_state = tf.placeholder(tf.float32, [None, state_shape])  # next state values (excluding image)
        self.post_img = tf.placeholder(tf.float32, [None, 64, 64, 1])  # next image
        self.Q_gradient =  tf.placeholder(tf.float32, [None, action_dim])  # gradient of current Q value w.r.t action (obtained from Critic)
        
        with tf.variable_scope(name + "actor"):
            # self.eval_net = self._build_network(self.state, self.img, "eval_net")  # build current network
            self.eval_net = self._build_network(self.state, "eval_net")  # build current network
            # target network predicts next action for Critic's target network, as part of actual Q value: current reward plus target Q value corresponding to current state and predicted next action
            # self.target_net = self._build_network(self.post_state, self.post_img, "target_net")  # build target network
            self.target_net = self._build_network(self.post_state, "target_net")  # build target network
        
        self.eval_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + "actor/eval_net")  # get current network parameters
        self.target_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + "actor/target_net")  # get target network parameters
        
        # use negative Q gradient to guide gradient ascent
        self.policy_gradient = tf.gradients(ys=self.eval_net, xs=self.eval_param, grad_ys=-self.Q_gradient)  # deterministic policy gradient
        self.train_step = tf.train.AdamOptimizer(lr).apply_gradients(zip(self.policy_gradient, self.eval_param))
        
        self.update_ops = self._update_target_net_op()  # assign current network parameters to target network
        
    # def _build_network(self, X, image, scope):
    #     with tf.variable_scope(scope):
    #         init_w1 = tf.truncated_normal_initializer(0., 3e-4)  # generate random numbers from normal distribution (but within two standard deviations, others truncated)
    #         init_w2 = tf.random_uniform_initializer(-0.05, 0.05)  # also generate random numbers from normal distribution
    #
    #         conv1 = tf.layers.conv2d(image, 32, [5,5], strides=[2,2], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)
    #         pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    #         conv2 = tf.layers.conv2d(pool1, 32, [5,5], strides=[1,1], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)
    #         pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
    #         conv3 = tf.layers.conv2d(pool2, 32, [5,5], strides=[1,1], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)
    #         pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)
    #         flatten = tf.layers.flatten(pool3) # shape(None, 4*4*32)
    #         concat = tf.concat([flatten, X], 1)
    #
    #         fc1 = tf.layers.dense(inputs=concat, units=200, activation=tf.nn.relu, kernel_initializer=init_w2)
    #         fc2 = tf.layers.dense(inputs=fc1, units=200, activation=tf.nn.relu, kernel_initializer=init_w2)
    #         fc3 = tf.layers.dense(inputs=fc2, units=200, activation=tf.nn.relu, kernel_initializer=init_w2)
    #         action_normal = tf.layers.dense(inputs=fc3, units=self.action_dim, activation=tf.nn.tanh, kernel_initializer=init_w2)
    #         action = tf.multiply(action_normal, self.action_bound)
    #     return action
    def _build_network(self, X, scope):
        with tf.variable_scope(scope):
            init_w1 = tf.random_uniform_initializer(-0.05, 0.05)  # also generate random numbers from uniform distribution
            concat = tf.concat([X], 1)

            fc1 = tf.layers.dense(inputs=concat, units=200, activation=tf.nn.relu, kernel_initializer=init_w1)
            fc2 = tf.layers.dense(inputs=fc1, units=200, activation=tf.nn.relu, kernel_initializer=init_w1)
            fc3 = tf.layers.dense(inputs=fc2, units=200, activation=tf.nn.relu, kernel_initializer=init_w1)
            action_normal = tf.layers.dense(inputs=fc3, units=self.action_dim, activation=tf.nn.tanh, kernel_initializer=init_w1)
            action = tf.multiply(action_normal, self.action_bound)
        return action
    # current network computes actual action to execute
    def act(self, state):
        # img = state[0]
        # dstate = state[1:4]
        # img = np.reshape(img, [1, 64, 64, 1])
        state = np.reshape(state, [1, self.state_shape])
        action = self.sess.run(self.eval_net, feed_dict={self.state:state})[0]
        return action
    # current network predicts next action to execute
    def predict_action(self, states):
        # imgs, dstates = self._seperate_image(states)
        pred_actions = self.sess.run(self.eval_net, feed_dict={self.state:states})
        return pred_actions
    # target network used to compute next action
    def target_action(self, post_states):
        # imgs, dstates = self._seperate_image(post_states)
        actions = self.sess.run(self.target_net, feed_dict={self.post_state:post_states})
        return actions
    # train current network parameters
    def train(self, Q_gradient, states):
        # imgs, dstates = self._seperate_image(states)
        self.sess.run(self.train_step, feed_dict={self.state:states, self.Q_gradient:Q_gradient})
    # assign current network parameters to target network parameters
    def _update_target_net_op(self):
        ops = [tf.assign(dest_var, (1-self.tau) * dest_var + self.tau * src_var)
               for dest_var, src_var in zip(self.target_param, self.eval_param)]
        return ops
    # separate image data from state
    def _seperate_image(self, states):
        images = np.array([state[0] for state in states])
        dstates = np.array([state[1] for state in states])
        return images, dstates