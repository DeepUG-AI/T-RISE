import numpy as np
from .sum_tree import SumTree

class Memory(object):
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity, transition_len):
        self.tree = SumTree(capacity)
        self.transition_len = int(transition_len)

    def store(self, transition):
        transition = np.asarray(transition, dtype=np.float32).flatten()
        if transition.shape[0] != self.transition_len:
            raise ValueError(f"Transition length mismatch: got {transition.shape[0]}, expect {self.transition_len}")
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx = np.empty((n,), dtype=np.int32)
        b_memory = np.empty((n, self.transition_len), dtype=np.float32)
        ISWeights = np.empty((n, 1), dtype=np.float32)
        pri_seg = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors = np.asarray(abs_errors).flatten()
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
