import sys
sys.path.insert(0, './model005')

import numpy as np
import time
import random
import torch
import math

from torch.utils.data import DataLoader

from environment import WRSNEnv

from utils import WRSNDataset
from utils import DrlParameters as dp, WrsnParameters as wp
from utils import NetworkInput, Point
from utils import gen_cgrg, dist
from utils import device
from main import validate
from random_strategy import random_decision_maker

class GSA:
    def __init__(self):
        self.scheme = []
        self.nb_agent_default = 60
        self.nb_agent = 60
        self.t_max_default = 100
        self.t_max = 100
        self.g_0 = 100
        self.beta = 20
        self.epsilon = 10 ** -3
    
    def reset(self):
        self.scheme = []

    def init_population(self, requests):
        pop = []
        for i in range(self.nb_agent):
            scheme = 2 * np.random.rand(len(requests)) - 1
            velocity = 0.0 * np.random.rand(len(requests))
            fitness = self.fitness(scheme)
            pop.append({"scheme": scheme, "velocity": velocity, "fitness": fitness})
        return pop

    def g(self, t):
        return self.g_0 * math.exp(-self.beta * t / self.t_max)

    def m(self, population, best, worst):
        m = []
        for i, individual in enumerate(population):
            m.append((individual["fitness"] - worst + self.epsilon) / (best - worst + self.epsilon))
        return np.array([item / sum(m) for item in m])

    def best_worst(self, population):
        id_best = -1
        best = float("inf")
        worst = float("-inf")
        for id_individual, individual in enumerate(population):
            if individual["fitness"] < best:
                id_best = id_individual
                best = individual["fitness"]
            if individual["fitness"] > worst:
                worst = individual["fitness"]
        return best, worst, id_best

    def fitness(self, scheme):
        order = np.argsort(scheme)
        latency = []
        time = 0
        mc_loc = Point(self.mc_state[0], self.mc_state[1])
        for i in order:
            sn_point = Point(self.sn_state[i, 0], self.sn_state[i, 1])
            d_mc_i = dist(mc_loc, sn_point)
            t_mc_i = d_mc_i / self.mc_state[6]
            time += t_mc_i.item()
            t_charge_i = (self.sn_state[i, 2] - max(0, self.sn_state[i, 4] - self.sn_state[i, 5] * time)) / \
                    (self.mc_state[5] - self.sn_state[i, 5])
            time += t_charge_i.item()
            latency.append(time)
            mc_loc = sn_point

        return sum(latency) / len(latency)

    def schedule(self, requests, mc_state, depot_state, sn_state, mask):
        self.mc_state = mc_state
        self.depot_state = depot_state
        self.sn_state = sn_state[torch.nonzero(mask[1:]).squeeze()]
        self.mask = mask
        if len(requests) < 10:
            self.nb_agent = min(self.nb_agent_default, math.factorial(len(requests)//2+1))
            self.t_max = min(self.t_max_default, math.factorial(len(requests)//2+1))
        else:
            self.nb_agent = self.nb_agent_default
            self.t_max = self.t_max_default

        pop = self.init_population(requests)
        t = -1
    
        while t < self.t_max:
            t += 1
            best, worst, id_best = self.best_worst(pop)
            m = self.m(pop, best, worst)
            for i, _ in enumerate(pop):
                f_i = []
                for j, _ in enumerate(pop):
                    if j == i:
                        continue
                    f_j = self.g(t) * m[i] * m[j] / (
                            math.dist(pop[i]["scheme"], pop[j]["scheme"]) + self.epsilon) * (
                                  pop[i]["scheme"] - pop[j]["scheme"])
                    f_i.append(np.random.random() * f_j)
                #  update f_i
                f_i = np.asarray(f_i)
                f_i = np.sum(f_i, 0)
                #  update a_i
                a_i = f_i / m[i]
                #  update velocity
                pop[i]["velocity"] = np.random.rand(len(a_i)) * pop[i]["velocity"] + a_i
                #  update scheme
                pop[i]["scheme"] = pop[i]["scheme"] + pop[i]["velocity"]
                pop[i]["fitness"] = self.fitness(pop[i]["scheme"])
        best, worst, id_best = self.best_worst(pop)
        best_scheme = pop[id_best]["scheme"]
        order = np.argsort(best_scheme)
        self.scheme = [requests[i] for i in order]   

def gsa_decision_maker(mc_state, depot_state, sn_state, mask, gsa):
    mask_ = mask.clone()
    mask__ = mask.clone()
    mask_[0] = 0.0
    n = len(sn_state)
    d_mc = torch.zeros(n+1)
    no_requests = 0

    for i in range(0, n):
        d_mc_i = dist(Point(mc_state[0], mc_state[1]),
                      Point(sn_state[i, 0], sn_state[i, 1]))
        d_mc[i+1] = d_mc_i

        t_mc_i = d_mc_i / mc_state[6]
        d_i_bs = dist(Point(sn_state[i, 0], sn_state[i, 1]),
                      Point(**wp.depot))
        t_charge_i = (sn_state[i, 2] - sn_state[i, 4] + sn_state[i, 5] * t_mc_i) / \
                    (mc_state[5] - sn_state[i, 5])
        if mc_state[2] - mc_state[4] * d_mc_i - \
            (sn_state[i, 2] - sn_state[i, 4] + sn_state[i, 5] * (t_mc_i + t_charge_i)) \
            - mc_state[4] * d_i_bs < 0:
            mask_[i+1] = 0.0
            mask__[i+1] = 0.0
        if sn_state[i, 4] > sn_state[i, 2] * 0.4:
            mask_[i+1] = 0.0
        else:
            no_requests += 1

    if no_requests == 0: 
        return np.random.choice(np.nonzero(mask__.cpu().numpy())[0]), 0.0
    
    requests = mask_.cpu().numpy().nonzero()[0]

    if len(requests) == 0:
        return 0, 0
    if len(requests) == 1:
        return requests[0], 1.0
    if not gsa.scheme:
        gsa.schedule(requests, mc_state, depot_state, sn_state, mask_)
    
    next_loc = gsa.scheme[0]
    del gsa.scheme[0]

    return next_loc, 1.0


def run_gsa(data_loader, name, save_dir, wp, max_step=1000):
    def gsa_reset(gsa):
        gsa.reset()

    gsa = GSA()
    return validate(data_loader, gsa_decision_maker, (gsa,), wp=wp, 
                    normalize=False, max_step=max_step, on_episode_begin=gsa_reset)

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)
    seed=123
    torch.manual_seed(seed-1)
    np.random.seed(seed-2)
    dataset = WRSNDataset(20, 10, 1, 1)
    wp.k_bit = 6000000
    data_loader = DataLoader(dataset, 1, False, num_workers=0)
    gsa = GSA()
    run_gsa(data_loader, 'gsa', './', wp)
    # validate(data_loader, gsa_decision_maker, (gsa, ), render=False, verbose=True, normalize=False)
    # validate(data_loader, random_decision_maker, render=False, verbose=True, normalize=False)
