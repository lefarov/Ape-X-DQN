import pickle
import torch
import numpy as np

from collections import OrderedDict


def model_state_bsize(model_state):
    return sum([v.element_size() * v.nelement() for v in model_state.values()])


def model_state_nelements(model_state):
    return sum([v.nelement() for v in model_state.values()])


def model_state_to_ndarray(model_state):
    return torch.cat([v.flatten() for v in model_state.values()]).numpy()


def ndarray_to_model_state(model_array, model_state):
    loaded_state = OrderedDict()

    idx = 0
    for key, param in model_state.items():
        loaded_state[key] = torch.from_numpy(
            model_array[idx : idx + param.nelement()]
        ).view(param.size())
        idx += param.nelement()

    return loaded_state


def model_states_equal(state1, state2):
    equal = True
    for (k1, t1), (k2, t2) in zip(state1.items(), state2.items()):
        equal = equal and torch.equal(t1, t2)
        equal = equal and k1 == k2

    return equal


def transition_buffer_to_bytes(buffer):
    return "".join(list(map(lambda t: pickle.dumps(t), buffer)))


def bytes_to_transition_buffer(bts, transition_len):
    transitions = []
    for idx in range(0, len(bts), transition_len):
        transitions.append(pickle.loads(bts[idx : idx + transition_len]))


def tranition_buffers_equal(buffer1, buffer2):
    equal = True
    for t1, t2 in zip(buffer1, buffer2):
        for v1, v2 in zip(t1, t2):
            if isinstance(v1, np.ndarray):
                equal = equal and np.array_equal(v1, v2)
            else:
                equal = equal and v1 == v2

    return equal
