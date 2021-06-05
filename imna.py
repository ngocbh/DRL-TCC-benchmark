import sys
sys.path.insert(0, './model006')

import numpy as np
import time
import random
import torch

from torch.utils.data import DataLoader

from environment import WRSNEnv

from utils import WRSNDataset
from utils import DrlParameters as dp, WrsnParameters as wp
from utils import NetworkInput, Point
from utils import gen_cgrg, dist
from utils import device
from main import validate
from random_strategy import random_decision_maker

def imna_decision_maker(mc_state, depot_state, sn_state, mask):

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

    latency = sn_state[:, 4] / sn_state[:, 5]
    w = torch.zeros(n, n).to(device)
    Z = torch.zeros(n+1)
    t_charge = torch.zeros(n+1)

    for i in range(0, n):
        t_mc_i = dist(Point(mc_state[0], mc_state[1]),
                          Point(sn_state[i, 0], sn_state[i, 1])) / mc_state[6]
        t_charge_i = (sn_state[i, 2] - sn_state[i, 4] + sn_state[i, 5] * t_mc_i) / \
                    (mc_state[5] - sn_state[i, 5])
        t_charge[i+1] = t_mc_i + t_charge_i
        for j in range(0, n):
            t_i_j = dist(Point(sn_state[i, 0], sn_state[i, 1]),
                          Point(sn_state[j, 0], sn_state[j, 1])) / mc_state[6]
            w[i, j] = t_mc_i + t_charge_i + t_i_j
        Z[i+1] = (latency <= w[i]).sum()

    valid_Z_idx = mask_.cpu().numpy().nonzero()[0]

    if len(valid_Z_idx) > 0:
        weight = Z * 1e8 + t_charge

        action = valid_Z_idx[weight[valid_Z_idx].argmin()]
        return action.item(), weight[valid_Z_idx].min()
    else:
        return 0, 1.0

def run_imna(data_loader, name, save_dir, wp, max_step=1000):
    return validate(data_loader, imna_decision_maker, wp=wp, normalize=False, max_step=max_step)

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)
    seed=123
    torch.manual_seed(seed-1)
    np.random.seed(seed-2)
    dataset = WRSNDataset(20, 10, 1, 1)
    wp.k_bit = 20000000
    data_loader = DataLoader(dataset, 1, False, num_workers=0)
    validate(data_loader, imna_decision_maker, render=False, verbose=True, normalize=False)
    # validate(data_loader, random_decision_maker, render=False, verbose=True, normalize=False)
