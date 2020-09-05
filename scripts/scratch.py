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
    def __init__(self, mpi_comm, buff_size):
        self.communicator = mpi_comm
        self.local_buffer = []
        self.size = buff_size

        # Create persistant send communication
        # self.req = self.communicator.Send_init(self.local_buffer, dest=0, tag=11)
        self.req = MPI.Request()

    def run(self):
        for i in range(1000000):
            self.local_buffer.append(
                Transition(
                    np.random.normal(0.0, 0.5, size=(10,)),
                    np.random.normal(0.0, 0.5, size=(2,)),
                    0.7,
                    0.9,
                    np.random.normal(0.0, 0.5, size=(2,)),
                )
            )
            self.local_buffer = self.local_buffer[-self.size :]

            if i % 100 == 0:
                print(
                    f"worker: {comm.Get_rank()}/{comm.Get_size()}, "
                    f"iteration: {i}, t:{time.time() - start_time}, "
                    f"data length: {len(self.local_buffer)}"
                )

                # Check if previous send is completed
                if not MPI.Request.Test(self.req):
                    # If nto wait for completion
                    print(
                        f"worker: {comm.Get_rank()}/{comm.Get_size()}, "
                        f"previous send was not completed"
                    )
                    self.req.wait()

                self.req = self.communicator.isend(self.local_buffer, dest=0, tag=11)


class Learner(object):
    def __init__(self, mpi_comm, buff_size):
        self.communicator = mpi_comm
        self.replay_buffer = []
        self.size = buff_size

        # Create persistant receive communication
        # self.req = self.communicator.Recv_init(source=1, tag=11)

    def learn(self):
        for i in range(1000):
            if i % 10 == 0:
                print(
                    f"learner: training iteration: {i}, t:{time.time() - start_time}, "
                    f"data length: {len(self.replay_buffer)}"
                )
                # self.req = self.communicator.irecv(source=1, tag=11)
                reqs = [
                    self.communicator.irecv(source=w, tag=11)
                    for w in range(1, self.communicator.Get_size())
                ]

                data = MPI.Request.waitall(reqs)

                print(
                    f"learner: received data type: {type(data)}, len: {len(data)}, "
                    # f"sample: {data}"
                )
                self.replay_buffer.append(data)
                self.replay_buffer = self.replay_buffer[-self.size :]

            time.sleep(1)


# %%
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# %%
# rma_window = MPI.Win.Create()

if rank != 0:
    worker = Worker(comm, 100)
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
    learner = Learner(comm, 10000)
    learner.learn()
    # data = np.empty(10, dtype=np.int)
    # req = comm.Recv_init(data, source=1, tag=11)
    # req = comm.irecv(source=1, tag=11)
    # time.sleep(10)
    # req.Start()
    # data = req.wait()
    # print(type(data))
    # print(f"process {rank}/{size}: gathered data: {data}")
