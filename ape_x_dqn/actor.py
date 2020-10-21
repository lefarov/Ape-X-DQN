import abc
import torch
import random
import numpy as np
import typing as typ
import torch.multiprocessing as mp

from collections import namedtuple, deque
from dataclasses import dataclass
from mpi4py import MPI

from ape_x_dqn.utilities import (
    transition_buffer_to_bytes,
    ndarray_to_model_state,
    model_state_nelements,
)

# from ape_x_dqn.duelling_network import DuellingDQN
# from ape_x_dqn.env import make_local_env

# import cv2

# Transition = namedtuple(
#     "Transition",
#     ["S", "A", "R", "Q", "S_next", "Q_next"],
# )


@dataclass
class Transition:
    S: typ.Any
    A: typ.Any
    R: float
    Q: float
    S_next: typ.Any
    Q_next: float


""" N Step Transition with accumulated discounted return

S:          (Any):      state at time t
A:          (int):      action at time t
R_nstep:    (float):    cummulative discount reward after N time steps
Q:          (flaot):    Q(S, A)
S_nstep:    (Any):      state after N time steps
Q_nstep:    (float):    max(a) Q(S_nstep, a)
key:        (str):      unique identifier of transition for priority dictionary

Q and Q_nstep are used only for theinitial priorities computation
"""
NStepTransition = namedtuple(
    "NStepTransition",
    ["S", "A", "R_nstep", "Q", "S_nstep", "Q_nstep", "key"],
)

# @dataclass
# class NStepTransition:
#     S: t.Any
#     A: t.Any
#     R_nstep: float
#     Q: float
#     S_nstep: t.Any
#     Q_nstep: float
#     key: str


class ExperienceBuffer(object):
    def __init__(self, nstep_buffer, nstep_capacity, n_steps, gamma, actor_id):
        self.__nstep_buffer = nstep_buffer
        self.nstep_capacity = nstep_capacity
        self.nstep_idx = 0

        # Local single step transition buffer
        self.local_sstep_buffer = deque(maxlen=n_steps)

        self.n_steps = n_steps
        self.gamma = gamma
        self.id = actor_id

        # Used to compute the unique n-step transition key
        self.nstep_transition_num = 0

    @property
    def nstep_buffer(self):
        return self.__nstep_buffer

    @property
    def nstep_idx(self):
        return self.__nstep_idx

    @nstep_idx.setter
    def nstep_idx(self, idx):
        self.__nstep_idx = idx % self.nstep_capacity

    @property
    def size(self):
        return len(self.local_sstep_buffer)

    def aggregate_return(self):
        for i, transition in enumerate(reversed(self.local_sstep_buffer)):
            discount = self.gamma ** i if i > 0 else 0
            transition.R += discount * self.local_sstep_buffer[-1].R

    def add(self, transition: Transition, done: bool) -> None:
        # Add transition to the local buffer of single transitions
        self.local_sstep_buffer.append(transition)
        # Agrreate the discounted rewards to the head of the local buffer
        self.aggregate_return()

        if self.size == self.n_steps or done:
            # Construct the N step transitions
            head = self.local_sstep_buffer.popleft()
            self.nstep_buffer[self.nstep_idx] = NStepTransition(
                head.S,
                head.A,
                head.R,
                head.Q,
                transition.S_next,
                transition.Q_next,
                f"{self.id}:{self.nstep_transition_num}",
            )

            self.nstep_idx += 1
            self.nstep_transition_num += 1

            if done:
                # Clear the local buffer
                self.local_sstep_buffer.clear()


class Worker(abc.ABC):
    def __init__(
        self,
        q_net: torch.nn.Module,
        policy: typ.Callable[[typ.Any, int], int],
        env,
        capacity,
        gamma,
        model_update_t=200,
        transitions_nsteps=1,
        transitions_send_t=200,
        total_time_steps=1000,
    ):
        self.id = 0
        self.q_net = q_net
        self.policy = policy
        self.env = env
        self.capacity

        self.mutable_experience_buffer = [None] * self.capacity
        self.nstep_experience = ExperienceBuffer(
            self.mutable_experience_buffer,
            capacity,
            transitions_nsteps,
            gamma,
            self.id,
        )

        self.gamma = gamma
        self.nsteps = transitions_nsteps

        self.model_update_t = model_update_t
        self.trans_send_t = transitions_send_t
        self.total_t = total_time_steps

        # Get initial model parameters
        self._update_model()

    def compute_priorities(self, transitions: typ.List[NStepTransition]):
        # Conver list of NStepTransitoins to the NStepTransitoin with the list members
        n_step_transitions = NStepTransition(*zip(*transitions))

        q = np.array(n_step_transitions.Q)
        r_nstep = np.array(n_step_transitions.R_nstep)
        q_nstep = np.array(n_step_transitions.Q_nstep)

        # Compute TD error (no Double-Q)
        target = r_nstep + (self.gamma ** self.nsteps) * q_nstep
        td_error_nstep = abs(q - target)

        return {k: v for k in n_step_transitions.key for v in td_error_nstep}

    @torch.no_grad()
    def get_q(self, obs):
        return self.q_net(obs)

    def run(self):
        # Reset environment and compute initial Q values
        obs = self.env.reset()
        q_vals = self.get_q(obs)

        for t in range(self.tsteps):
            # Compute actrions from Q values
            act = self.policy(q_vals, t)

            # Step throught the environment and add transitiont to experience
            obs_next, rew, done, info = self.env.step(act)
            q_vals_next = self.get_q(obs_next)

            self.nstep_experience.add(
                Transition(
                    obs, act, rew, q_vals[act], obs_next, torch.max(q_vals_next)
                ),
                done,
            )

            # If done, reset the environment
            if done:
                obs_next = self.env.reset()
                q_vals_next = self.get_q(obs_next)

            # Step forward
            obs = obs_next
            q_vals = q_vals_next

            # Send transitions
            if t % self.trans_send_t == 0 and t > 0:
                self._send_transitions()

            # Update model
            if t % self.model_update_t == 0 and t > 0:
                self._update_model()

    @abc.abstractmethod
    def _update_model(self):
        pass

    @abc.abstractmethod
    def _send_transitions(self):
        pass


class MPIWorker(Worker):
    def __init__(
        self,
        mpi_comm,
        model_window,
        transition_bsize,
        transitions_window,
        transitions_idx_window,
        capacity_total,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.blen_total = capacity_total * transition_bsize

        self.model_window = model_window
        self.model_state = self.q_net.state_dict()
        self.model_array = np.zeros(
            model_state_nelements(self.model_state), dtype=np.float32
        )

        self.trans_window = transitions_window
        self.trans_idx_window = transitions_idx_window

    def _send_transitions(self):
        # Lock the RMA window for replay buffer on Learner process.
        # Idx window is not locked explicitely, but it's accessed
        # only within the replay window lock.
        self.trans_window.Lock(rank=0)

        # Read current index of total replay buffer
        idx = int.from_bytes(self.trans_idx_window, byteorder="big")

        # Pickle the transitions
        # If N Step buffer is not complete yet
        if self.mutable_experience_buffer[self.nstep_experience.nstep_idx] is None:
            # Send dat till this index
            barray = transition_buffer_to_bytes(
                self.buffer[: self.nstep_experience.nstep_idx]
            )
        else:
            # Send all data otherwise
            barray = transition_buffer_to_bytes(self.buffer)

        # If Learner Replay has enough space
        if idx + len(barray) < self.blen_total:
            # Write data in one block
            self.trans_window.Put(
                barray, target_rank=0, target=(idx, len(barray), MPI.BYTE)
            )
            # Increment the index
            idx += len(barray)

        else:
            # Write data to the end of the buffer
            n = self.blen_total % idx
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

        # Unlock the RMA window on Learner process
        self.trans_window.Unlock(rank=0)

    def _update_model(self):
        # Lock the RMA window for model parameter on Learner process and read parameters
        self.model_window.Lock(rank=0)
        self.model_window.Get(self.model_array, target_rank=0)
        self.model_window.Unlock(rank=0)

        # Load received parameters to the model
        self.model.load_state_dict(
            ndarray_to_model_state(self.model_array, self.model_state)
        )


class TorchDistWorker(Worker, mp.Process):
    def __init__(
        self,
        model_shared_state,
        transitions_shared_mem,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_state = model_shared_state
        self.trans_mem = transitions_shared_mem

    def _send_transitions(self):
        # Calculate the priorities for experience
        priorities = self.compute_priorities(self.mutable_experience_buffer)
        # Send the experience to the global replay memory
        self.trans_mem.put([priorities, self.mutable_experience_buffer])

    def _update_model(self):
        self.q_net.load_state_dict(self.model_state["q_state_dict"])
