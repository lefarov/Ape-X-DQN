# %% Imports
import time
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

from mpi4py import MPI
from collections import namedtuple, OrderedDict


# %% Helper functions


def model_state_bsize(model_state):
    return sum([v.element_size() * v.nelement() for v in model_state.values()])


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


def states_equal(state1, state2):
    equal = True
    for (k1, t1), (k2, t2) in zip(state1.items(), state2.items()):
        equal = equal and torch.equal(t1, t2)
        equal = equal and k1 == k2

    return equal


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


# %%
class Worker(object):
    def __init__(self, mpi_comm, buffer_size, buffer_state, policy_state, n_steps=100):
        self.comm = mpi_comm
        self.size = buffer_size
        self.buffer = []
        self.policy = policy_state

        self.step_n = n_steps
        self.step_i = 0

        # Receive policy (blocking)
        self.policy = self.comm.bcast(self.policy, root=0)
        print(
            f"worker: {self.comm.Get_rank()}/{self.comm.Get_size()}, "
            f"policy: {self.policy['W']}, "
        )
        # Initialize empty request for sending rollouts
        self.req = MPI.Request()

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
                self.policy = self.comm.bcast(self.policy, root=0)
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
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

_POLICY_BCST_STATE = dict()
_BUFFER_GTHR_STATE = list()

temp_policy = torch.rand((256, 256))
policy_window = MPI.Win.Allocate(
    temp_policy.element_size() * temp_policy.nelement(), comm=comm
)

if rank != 0:
    # policy = torch.ones_like(temp_policy).numpy().flatten()
    policy = np.ones(temp_policy.nelement(), dtype=np.float32)
    win_array = np.frombuffer(policy_window, dtype=np.float32)
    # win_array[0] = 7.
    print(f"worker: {comm.Get_rank()}/{comm.Get_size()}, initial policy {policy}")
    time.sleep(2)
    policy_window.Lock(rank=0)
    # window.Flush(rank=0)
    policy_window.Get([policy, 2], target_rank=0)
    # policy[:2] = win_array[:2]
    policy_window.Unlock(rank=0)
    print(f"worker: {comm.Get_rank()}/{comm.Get_size()}, loaded policy {policy}")

    # worker = Worker(comm, 100, _BUFFER_GTHR_STATE, _POLICY_BCST_STATE)
    # worker.run()

if rank == 0:
    policy = torch.ones_like(temp_policy).numpy().flatten() * 5.0
    win_array = np.frombuffer(policy_window, dtype=np.float32)
    print(f"worker: {comm.Get_rank()}/{comm.Get_size()}, initial policy {policy}")
    policy_window.Lock(rank=0)
    print(win_array)
    policy_window.Put(policy, target_rank=0)
    # win_array[:2] = policy[:2]
    # window.Flush(rank=1)
    # window.Sync()
    print(win_array)
    policy_window.Unlock(rank=0)

    # learner = Learner(comm, 10000, _BUFFER_GTHR_STATE, _POLICY_BCST_STATE)
    # learner.learn()
