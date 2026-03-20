import copy
import numpy as np

class DelayCompensator:
    def __init__(self, delay_steps=1, delay_comp_alpha=0.1):
        self.delay_steps = delay_steps
        self.delay_comp_alpha = delay_comp_alpha

    def apply(self, state_buffer, state_raw, enable_delay_model=True):
        state_buffer.append(copy.deepcopy(state_raw))
        if not enable_delay_model:
            return state_raw
        if len(state_buffer) <= self.delay_steps:
            return state_raw
        delayed_state = copy.deepcopy(state_buffer[-1 - self.delay_steps])
        if len(state_buffer) > self.delay_steps + 1:
            prev_delayed_state = copy.deepcopy(state_buffer[-2 - self.delay_steps])
            comp_state = copy.deepcopy(delayed_state)
            if comp_state[0] != -1 and prev_delayed_state[0] != -1:
                comp_state[0] = delayed_state[0] + self.delay_comp_alpha * (delayed_state[0] - prev_delayed_state[0])
                comp_state[0] = max(comp_state[0], 0.0)
            if comp_state[1] != -1 and prev_delayed_state[1] != -1:
                comp_state[1] = delayed_state[1] + 0.10 * (delayed_state[1] - prev_delayed_state[1])
                comp_state[1] = float(np.clip(comp_state[1], -180.0, 180.0))
            return comp_state
        return delayed_state
