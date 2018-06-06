#!/usr/bin/env python
import torch
import torch.multiprocessing as mp
import json
from replay import ReplayMemory
from actor import Actor
from duelling_network import DuellingDQN
from argparse import ArgumentParser

arg_parser = ArgumentParser(prog="main.py")
arg_parser.add_argument("--params-file", default="parameters.json", type=str,
                    help="Path to json file defining the parameters for the Actor, Learner and Replay memory",
                    metavar="PARAMSFILE")
args = arg_parser.parse_args()


if __name__ =="__main__":
    params = json.load(open(args.params_file, 'r'))
    env_conf = params['env_conf']
    actor_params = params["Actor"]
    learner_params = params["Learner"]
    replay_params = params["Replay_Memory"]

    dummy_q = DuellingDQN(tuple(env_conf['state_shape']), env_conf['action_dim'])
    mp_manager = mp.Manager()
    shared_state = mp_manager.dict()
    shared_state["Q_state_dict"] = dummy_q.state_dict()
    shared_replay_mem = mp_manager.Queue()
    #  TODO: Start Actors in separate proc
    actor = Actor(1, env_conf, shared_state, shared_mem, actor_params)
    actor_procs = mp.Process(target=actor.gather_experience, args=(1110,))
    actor_procs.start()

    # TODO: Run a routine in a separate proc to fetch/pre-fetch shared_replay_mem onto the ReplayBuffer for learner's use
    actor_procs.join()

    print("Main: replay_mem.size:", shared_replay_mem.qsize())