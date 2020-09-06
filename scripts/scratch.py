# %%
# import os
# import json
import time
import numpy as np

from mpi4py import MPI
from collections import namedtuple

# %%
start_time = time.time()

# %%
Transition = namedtuple("Transition", ["S", "A", "R", "Gamma", "Q"])


class Worker(object):
    def __init__(
        self, mpi_comm, buffer_size, buffer_state, policy_state, n_steps=100
    ):
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
    def __init__(
        self, mpi_comm, buffer_size, buffer_state, policy_state, n_steps=10
    ):
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


# %%
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# %%
# rma_window = MPI.Win.Create()

_POLICY_BCST_STATE = dict()
_BUFFER_GTHR_STATE = list()

if rank != 0:
    worker = Worker(comm, 100, _BUFFER_GTHR_STATE, _POLICY_BCST_STATE)
    worker.run()
    # data = [
    #     Transition(
    #         np.random.normal(0.0, 0.5, size=(10,)),
    #         np.random.normal(0.0, 0.5, size=(2,)),
    #         0.7,
    #         0.9,
    #         np.random.normal(0.0, 0.5, size=(2,)),
    #     )
    #     for _ in range(10)
    # ]

    # buff = np.arange(10)

    # req = comm.Send_init(buff, dest=0, tag=11)
    # req = comm.isend(buff, dest=0, tag=11)
    # req.Start()

    # print("I'm doing some other stuff")
    # buff[0] = 100
    # print(f"process {rank}/{size}: gathered data: {buff}")
    # req.wait()

# Gather data (Blocking operation)
# data = comm.gather(data, root=0)

if rank == 0:
    learner = Learner(comm, 10000, _BUFFER_GTHR_STATE, _POLICY_BCST_STATE)
    learner.learn()
    # data = np.empty(10, dtype=np.int)
    # req = comm.Recv_init(data, source=1, tag=11)
    # req = comm.irecv(source=1, tag=11)
    # time.sleep(10)
    # req.Start()
    # data = req.wait()
    # print(type(data))
    # print(f"process {rank}/{size}: gathered data: {data}")
