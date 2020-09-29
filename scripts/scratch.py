# %% Imports
import time
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
        buffer_size,
        model,
        model_window,
        transitions_window,
        n_steps=100,
    ):
        self.comm = mpi_comm

        # Local experiance buffer
        self.buffer = []
        self.size = buffer_size

        # RMA window for transitions
        self.trans_window = transitions_window

        # Model object and RMA window for model parameters
        self.model = model
        self.model_window = model_window

        # Model state dictionary and parameters buffer
        self.model_state = model.state_dict()
        self.model_array = np.zeros(
            model_state_nelements(self.model_state), dtype=np.float32
        )

        # Read initial model parameters from the Learner's RMA window
        self.model_window.Lock(rank=0)
        self.model_window.Get(self.model_array, target_rank=0)
        self.model_window.Unlock(rank=0)

        # Load initial parameters to the model
        self.model.load_state_dict(
            ndarray_to_model_state(self.model_array, self.model_state)
        )

        # Initialize empty request for sending rollouts
        self.req = MPI.Request()

        self.step_n = n_steps
        self.step_i = 0

    def run(self):
        # TODO: collect data while previous send is not completed.

        for i in range(1000000):
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

            # TODO: collect n data points per policy
            if self.step_i == self.step_n:
                print(
                    f"worker: {self.comm.Get_rank()}/{self.comm.Get_size()}, "
                    f"iteration: {i}, t:{time.time() - start_time}, "
                    f"data length: {len(self.buffer)}"
                )

                # Check if previous send is completed
                if not MPI.Request.Test(self.req):
                    # If nto wait for completion
                    print(
                        f"worker: {self.comm.Get_rank()}/{self.comm.Get_size()}, "
                        f"previous send was not completed"
                    )
                    self.req.wait()

                # Send rollouts to learner (non-blocking)
                # self.req = self.comm.isend(self.buffer, dest=0, tag=11)
                self.comm.gather(self.buffer, root=0)

                # Get the updated policy
                self.model = self.comm.bcast(self.model, root=0)
                print(
                    f"worker: {self.comm.Get_rank()}/{self.comm.Get_size()}, "
                    f"policy: {self.policy['W']}"
                )

                self.step_i = 0

            self.step_i += 1


class Learner(object):
    def __init__(self, mpi_comm, buffer_size, buffer_state, policy_state, n_steps=10):
        self.comm = mpi_comm
        self.size = buffer_size
        self.buffer = []
        self.policy = policy_state

        self.step_n = n_steps
        self.step_i = 0

        # Create persistant receive communication
        # self.req = self.communicator.Recv_init(source=1, tag=11)

        # Send initialized policy to all workers
        self.policy["W"] = 0.0
        self.comm.bcast(self.policy, root=0)

    def learn(self):
        # TODO: open policy recv request here and while not completed collect the rollouts
        for i in range(1000):

            if self.step_i == self.step_n:
                print(
                    f"learner: training iteration: {i}, t:{time.time() - start_time}, "
                    f"data length: {len(self.buffer)}"
                )

                # Received data from workers (non-blocking)
                # reqs = [
                #     self.comm.irecv(source=w, tag=11)
                #     for w in range(1, self.comm.Get_size())
                # ]

                # Gather the data from workers (blocking)
                data = self.comm.gather(self.buffer, root=0)

                # Send updated policy to all workers (blocking)
                self.comm.bcast(self.policy, root=0)

                # Wait untill all data is received
                # data = MPI.Request.waitall(reqs)

                # Postptocess the data
                print(
                    f"learner: received data type: {type(data)}, len: {len(data)}, "
                    f"my buffer len: {len(data[0])}"
                )
                self.buffer.append(data)
                self.buffer = self.buffer[-self.size :]

                self.step_i = 0

            # Time deelay for training
            time.sleep(1)
            self.policy["W"] += 1.0
            self.step_i += 1


# %% Main
_TEST_SIGNLE_TRANSFER = True

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

# Create array of transitions
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

counter_window = MPI.Win.Allocate(4, comm=comm)

if rank != 0:
    if _TEST_SIGNLE_TRANSFER:
        read_model(model_window, net)
        write_transitions(trans_window, trans)

        counter_window.Lock(rank=0)
        counter_window.Put((10).to_bytes(4, byteorder="big"), target_rank=0)
        counter_window.Unlock(rank=0)

    else:
        pass
    # worker = Worker(comm, 100, _BUFFER_GTHR_STATE, _POLICY_BCST_STATE)
    # worker.run()

if rank == 0:
    if _TEST_SIGNLE_TRANSFER:
        share_model(model_window, net)
        read_transitions(trans_window, trans, trans_bsize)

        # Wait for 2 seconds
        time.sleep(3)

        # Read model state from the Learner's window
        counter_window.Lock(rank=0)
        print(int.from_bytes(counter_window, byteorder="big"))
        counter_window.Unlock(rank=0)

    else:
        pass
    # learner = Learner(comm, 10000, _BUFFER_GTHR_STATE, _POLICY_BCST_STATE)
    # learner.learn()


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
