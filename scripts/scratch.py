# %% Imports
import time
import random
import pickle
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

from mpi4py import MPI
from collections import namedtuple, OrderedDict


# %% Helper functions
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
    return b"".join(list(map(lambda t: pickle.dumps(t), buffer)))


def bytes_to_transition_buffer(bts, transition_bsize):
    transitions = []
    for idx in range(0, len(bts), transition_bsize):
        transitions.append(pickle.loads(bts[idx : idx + transition_bsize]))

    return transitions


def tranition_buffers_equal(buffer1, buffer2):
    equal = True
    for t1, t2 in zip(buffer1, buffer2):
        for v1, v2 in zip(t1, t2):
            if isinstance(v1, np.ndarray):
                equal = equal and np.array_equal(v1, v2)
            else:
                equal = equal and v1 == v2

    return equal


# %% Testing functions
def share_model(window, net):
    # Write model state to the Learner window
    window.Lock(rank=0)
    window.Put(model_state_to_ndarray(net.state_dict()), target_rank=0)
    window.Unlock(rank=0)


def read_model(window, net):
    # Create local copy of the network and the weights buffer
    net_local = Net()
    state_array = np.zeros(
        model_state_nelements(net_local.state_dict()), dtype=np.float32
    )

    # Wait for 2 seconds
    time.sleep(1)

    # Read model state from the Learner's window
    window.Lock(rank=0)
    window.Get(state_array, target_rank=0)
    window.Unlock(rank=0)

    # Verify the correct weights
    loaded_state = ndarray_to_model_state(state_array, net_local.state_dict())
    print(
        f"worker: {comm.Get_rank()}/{comm.Get_size()}, "
        f"weights are identical: {model_states_equal(net.state_dict(), loaded_state)}"
    )


def write_transitions(window, trans):
    # Pickle transitions
    trans_bts = transition_buffer_to_bytes(trans)

    # Write transition bytes to the Learner window
    window.Lock(rank=0)
    window.Put(trans_bts, target_rank=0, target=(0, len(trans_bts), MPI.BYTE))
    window.Unlock(rank=0)


def read_transitions(window, trans, transition_bsize):
    # Wait for 2 seconds
    time.sleep(2)

    # Read transitions from the buffer
    trans_window.Lock(rank=0)
    buffer = bytearray(trans_window)
    trans_window.Unlock(rank=0)

    # Unpickle received transtions
    trans_restored = bytes_to_transition_buffer(buffer, transition_bsize)

    # Verify unpickled transtions
    print(
        f"worker: {comm.Get_rank()}/{comm.Get_size()}, "
        f"transition are identical: {tranition_buffers_equal(trans_restored, trans)}"
    )


# %% Network
start_time = time.time()
Transition = namedtuple("Transition", ["S", "A", "R", "Gamma", "Q"])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# %% Worker and learner classes
class Worker(object):
    def __init__(
        self,
        mpi_comm,
        model,
        model_window,
        transition_bsize,
        transitions_window,
        transitions_idx_window,
        transitions_send_t=200,
        model_update_t=200,
        local_replay_size=200,
        total_replay_size=500,
        total_time_steps=1000,
    ):
        self.comm = mpi_comm

        # Local experiance buffer
        self.buffer = []
        self.size = local_replay_size
        self.total_reaplay_size = total_replay_size
        self.trans_bsize = transition_bsize
        self.trans_send_t = transitions_send_t
        self.model_update_t = model_update_t
        self.total_t = total_time_steps

        # RMA window for transitions and idx
        self.trans_window = transitions_window
        self.idx_window = transitions_idx_window

        # Bytearray to load the remote transition index (int32)
        # TODO: no need to store this one
        self.idx_barray = bytearray(4)

        # Model object and RMA window for model parameters
        self.model = model
        self.model_window = model_window

        # Model state dictionary and parameters buffer
        self.model_state = model.state_dict()
        self.model_array = np.zeros(
            model_state_nelements(self.model_state), dtype=np.float32
        )

        # Read initial model parameters from the Learner's RMA window
        self._update_model()

    def _update_model(self):
        # Lock the RMA window and read model parameters
        self.model_window.Lock(rank=0)
        self.model_window.Get(self.model_array, target_rank=0)
        self.model_window.Unlock(rank=0)

        # Load received parameters to the model
        self.model.load_state_dict(
            ndarray_to_model_state(self.model_array, self.model_state)
        )

    def _send_transitions(self):
        # Lock the RMA window for replay buffer (we don't need lock for idx)
        self.trans_window.Lock(rank=0)

        # Read current index
        self.idx_window.Get(self.idx_barray, target_rank=0)
        idx = int.from_bytes(self.idx_barray, byteorder="big")

        # Pickle the transitions
        barray = transition_buffer_to_bytes(self.buffer)

        # If Learner Replay has enough space
        # TODO: move that to class properties
        blen = self.total_reaplay_size * self.trans_bsize
        if idx + len(barray) < blen:
            # Write data in one block
            self.trans_window.Put(
                barray, target_rank=0, target=(idx, len(barray), MPI.BYTE)
            )
            # Increment the index
            idx += len(barray)

        else:
            # Write data to the end of the buffer
            n = blen % idx
            self.trans_window.Put(
                barray[:n],
                target_rank=0,
                target=(idx, n, MPI.BYTE),
            )
            # And to the beginning
            self.trans_window.Put(
                barray[n:],
                target_rank=0,
                target=(0, len(barray) - n, MPI.BYTE),
            )
            # Update the index
            idx = len(barray) - n

        # Write index back to the Learner RMA
        self.idx_window.Put(idx.to_bytes(4, byteorder="big"), target_rank=0)

        # Unlock the RMA window
        self.trans_window.Unlock(rank=0)

    def run(self):
        for t in range(self.total_t):
            # Append the transition to the buffer
            self.buffer.append(
                Transition(
                    np.random.normal(0.0, 0.5, size=(10,)),
                    np.random.normal(0.0, 0.5, size=(2,)),
                    0.7,
                    0.9,
                    np.random.normal(0.0, 0.5, size=(2,)),
                )
            )
            self.buffer = self.buffer[-self.size :]

            # Send data to the learner every N steps
            if t % self.trans_send_t == 0 and t > 0:
                self._send_transitions()
                print(
                    f"worker: {self.comm.Get_rank()}/{self.comm.Get_size()}, "
                    f"iteration: {t}, t:{time.time() - start_time}, "
                    f"has written to index: {int.from_bytes(self.idx_barray, byteorder='big')}"
                )

            # Update model with the Learner weights
            if t % self.model_update_t == 0 and t > 0:
                self._update_model()
                print(
                    f"worker: {self.comm.Get_rank()}/{self.comm.Get_size()}, "
                    f"policy: {self.model.fc3.bias}"
                )


class Learner(object):
    def __init__(
        self,
        mpi_comm,
        model,
        model_window,
        transition_bsize,
        transitions_window,
        transitions_idx_window,
        transitions_sample_t=100,
        model_update_t=100,
        total_replay_size=500,
        total_time_steps=1000,
    ):
        self.comm = mpi_comm

        # Experiance buffer
        self.replay = []
        # self.barray = bytearray(total_replay_size * trans_bsize)

        self.size = total_replay_size
        self.trans_bsize = transition_bsize
        self.model_update_t = model_update_t
        self.trans_sample_t = transitions_sample_t
        self.total_t = total_time_steps

        # RMA window for Replay buffer and idx
        self.trans_window = transitions_window
        self.trans_barray_idx = 0
        self.trans_full_flag = False
        self.idx_window = transitions_idx_window

        # Initialize transtion index
        self.trans_window.Lock(rank=0)
        self.idx_window.Put(
            self.trans_barray_idx.to_bytes(4, byteorder="big"), target_rank=0
        )
        self.trans_window.Unlock(rank=0)

        # Model object and RMA window for model parameters
        self.model = model
        self.model_window = model_window

        # Initialize model and Write model state to the Learner window
        self.model.fc3.bais = torch.zeros_like(self.model.fc3.bias)
        self.model_window.Lock(rank=0)
        self.model_window.Put(
            model_state_to_ndarray(self.model.state_dict()), target_rank=0
        )
        self.model_window.Unlock(rank=0)

    def learn(self):
        for t in range(self.total_t):
            # Send data to the learner every N steps
            if t % self.trans_sample_t == 0 and t > 0:
                # Lock transition RMA window
                self.trans_window.Lock(rank=0)

                # Check if buffer was filled completely
                # TODO: rewrite this check
                self.trans_full_flag = self.trans_full_flag or (
                    int.from_bytes(self.idx_window, byteorder="big")
                    < self.trans_barray_idx
                    and bytearray(self.trans_window)[0] != 0
                )

                print(
                    f"learner: {self.comm.Get_rank()}/{self.comm.Get_size()}, "
                    f"iteration: {t}, t:{time.time() - start_time}, "
                    f"index: {int.from_bytes(self.idx_window, byteorder='big')}, "
                    f"flag: {self.trans_full_flag}"
                )

                if self.trans_full_flag:
                    # Buffer has been filled (sample normally)
                    trans_restored = bytes_to_transition_buffer(
                        bytearray(self.trans_window), self.trans_bsize
                    )

                else:
                    # Buffer is not filled yet
                    # Update index
                    self.trans_barray_idx = int.from_bytes(
                        self.idx_window, byteorder="big"
                    )

                    # Sample up to index
                    trans_restored = bytes_to_transition_buffer(
                        bytearray(self.trans_window)[: self.trans_barray_idx],
                        self.trans_bsize,
                    )

                if trans_restored:
                    trans_sampled = random.sample(trans_restored, 1)
                    print(
                        f"learner: {self.comm.Get_rank()}/{self.comm.Get_size()}, "
                        f"iteration: {t}, t:{time.time() - start_time}, "
                        f"sampled transition: {trans_sampled[0]}"
                    )

                self.trans_window.Unlock(rank=0)

            # Update model with the Learner weights
            if t % self.model_update_t == 0 and t > 0:
                self.model.fc3.bias = nn.Parameter(
                    torch.ones_like(self.model.fc3.bias) * t / self.model_update_t
                )
                self.model_window.Lock(rank=0)
                self.model_window.Put(
                    model_state_to_ndarray(self.model.state_dict()), target_rank=0
                )
                self.model_window.Unlock(rank=0)
                print(
                    f"learner: {self.comm.Get_rank()}/{self.comm.Get_size()}, "
                    f"policy: {self.model.fc3.bias}"
                )


# %% Main
_TEST_SIGNLE_TRANSFER = False

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

net = Net()
model_window = MPI.Win.Allocate(model_state_bsize(net.state_dict()), comm=comm)

# Assign zeros to the params of shared network
zero_state = net.state_dict()
for key, value in zero_state.items():
    zero_state[key] = torch.zeros_like(value)

net.load_state_dict(zero_state)

counter_window = MPI.Win.Allocate(4, comm=comm)

# Create array of transitions
if _TEST_SIGNLE_TRANSFER:
    trans = [
        Transition(
            np.ones(10) * i,
            f"s{i}",
            0.7,
            0.9,
            np.ones(2),
        )
        for i in range(2)
    ]

    trans_bsize = len(pickle.dumps(trans[0]))
    trans_window = MPI.Win.Allocate(trans_bsize * 2, comm=comm)

else:
    trans = Transition(
        np.random.normal(0.0, 0.5, size=(10,)),
        np.random.normal(0.0, 0.5, size=(2,)),
        0.7,
        0.9,
        np.random.normal(0.0, 0.5, size=(2,)),
    )
    trans_bsize = len(pickle.dumps(trans))
    trans_window = MPI.Win.Allocate(trans_bsize * 500, comm=comm)


if rank != 0:
    if _TEST_SIGNLE_TRANSFER:
        read_model(model_window, net)
        write_transitions(trans_window, trans)

        counter_window.Lock(rank=0)
        counter_window.Put((10).to_bytes(4, byteorder="big"), target_rank=0)
        counter_window.Unlock(rank=0)

    else:
        worker = Worker(
            comm,
            net,
            model_window,
            trans_bsize,
            trans_window,
            counter_window,
            total_replay_size=1100,
            total_time_steps=5000,
        )
        worker.run()

if rank == 0:
    if _TEST_SIGNLE_TRANSFER:
        share_model(model_window, net)
        read_transitions(trans_window, trans, trans_bsize)

        # Wait for 2 seconds
        time.sleep(3)

        # Read model state from the Learner's window
        counter_window.Lock(rank=0)
        buff = bytearray(4)
        counter_window.Get(buff, target_rank=0)
        print(int.from_bytes(buff, byteorder="big"))
        print(int.from_bytes(counter_window, byteorder="big"))
        counter_window.Unlock(rank=0)

    else:
        learner = Learner(
            comm,
            net,
            model_window,
            trans_bsize,
            trans_window,
            counter_window,
            total_replay_size=1100,
            total_time_steps=5000,
        )
        learner.learn()


# %%
# policy = torch.ones_like(temp_policy).numpy().flatten()
# policy = np.ones(temp_policy.nelement(), dtype=np.float32)
# win_array = np.frombuffer(policy_window, dtype=np.float32)
# win_array[0] = 7.
# print(f"worker: {comm.Get_rank()}/{comm.Get_size()}, initial policy {policy}")
# policy[:2] = win_array[:2]
# print(f"worker: {comm.Get_rank()}/{comm.Get_size()}, loaded policy {policy}")
# window.Flush(rank=0)

# _POLICY_BCST_STATE = dict()
# _BUFFER_GTHR_STATE = list()

# policy = torch.ones_like(temp_policy).numpy().flatten() * 5.0
# win_array = np.frombuffer(policy_window, dtype=np.float32)
# print(f"worker: {comm.Get_rank()}/{comm.Get_size()}, initial policy {policy}")
# win_array[:2] = policy[:2]
# window.Flush(rank=1)
# window.Sync()


# Check if previous send is completed
# if not MPI.Request.Test(self.req):
# # If nto wait for completion
# print(
#     f"worker: {self.comm.Get_rank()}/{self.comm.Get_size()}, "
#     f"previous send was not completed"
# )
# self.req.wait()

# # Send rollouts to learner (non-blocking)
# # self.req = self.comm.isend(self.buffer, dest=0, tag=11)
# self.comm.gather(self.buffer, root=0)

# # Get the updated policy
# self.model = self.comm.bcast(self.model, root=0)


# # Initialize empty request for sending rollouts
# self.req = MPI.Request()


# Create persistant receive communication
# # self.req = self.communicator.Recv_init(source=1, tag=11)

# # Send initialized policy to all workers
# self.policy["W"] = 0.0
# self.comm.bcast(self.policy, root=0)


# if self.t == self.step_n:
#     print(
#         f"learner: training iteration: {i}, t:{time.time() - start_time}, "
#         f"data length: {len(self.buffer)}"
#     )

#     # Received data from workers (non-blocking)
#     # reqs = [
#     #     self.comm.irecv(source=w, tag=11)
#     #     for w in range(1, self.comm.Get_size())
#     # ]

#     # Gather the data from workers (blocking)
#     data = self.comm.gather(self.buffer, root=0)

#     # Send updated policy to all workers (blocking)
#     self.comm.bcast(self.policy, root=0)

#     # Wait untill all data is received
#     # data = MPI.Request.waitall(reqs)

#     # Postptocess the data
#     print(
#         f"learner: received data type: {type(data)}, len: {len(data)}, "
#         f"my buffer len: {len(data[0])}"
#     )
#     self.buffer.append(data)
#     self.buffer = self.buffer[-self.size :]

# # Time deelay for training
# time.sleep(1)
# self.policy["W"] += 1.0
