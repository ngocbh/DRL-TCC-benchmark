import sys
sys.path.append('./drl')

import random_strategy
import main as drl
from utils import device, pdump, pload
from utils import WRSNDataset
from utils import WrsnParameters as wp, DrlParameters as dp
from model import MCActor
from environment import WRSNEnv
from ept_config import EptConfig as ec
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os



def run_drl(data_loader, name, save_dir):
    actor = MCActor(dp.MC_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)

    save_dir = os.path.join(save_dir, name)
    checkpoint = ec.checkpoint
    path = os.path.join(checkpoint, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))

    ret = drl.validate(data_loader, actor, render=False)

    return ret

def run_random(data_loader, name, save_dir):
    save_dir = os.path.join(save_dir, name)
    return random_strategy.validate(data_loader, save_dir)


solvers = {
    "drl": run_drl,
    "random": run_random
}



def plot_mean_std(x, data, xlabel, ylabel, title, save_dir, plot_std=True):
    plt.style.use('seaborn-white')

    fig, ax = plt.subplots()
    for name, (mean, std) in data.items():
        ax.plot(x, mean, label=name)
        if plot_std:
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.legend(frameon=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    plt.savefig(os.path.join(save_dir, f'{title}.png'), dpi=400)
    plt.close('all')
    

def run_ept_1_2(ept, seed=123, save_dir='results', rerun=False):
    def run_ept_1(save_dir):
        num_targets = ec.ept1.num_targets
        min_num_sensors = ec.ept1.min_num_sensors
        max_num_sensors = ec.ept1.max_num_sensors
        used_solvers = ec.ept1.solvers
        res = defaultdict(list)
        for num_sensors in range(min_num_sensors, max_num_sensors):
            test_data = WRSNDataset(num_sensors, num_targets, dp.test_size, seed)
            data_loader = DataLoader(test_data, 1, False, num_workers=0)
            for name, solver in solvers.items():
                if name in used_solvers:
                    ret = solver(data_loader, name, save_dir)
                    res[name].append((num_sensors, ret))

        pdump(res, 'data.pickle', save_dir)

    def run_ept_2(save_dir):
        num_targets = ec.ept2.num_targets
        num_sensors = ec.ept2.num_sensors
        min_prob = ec.ept2.min_prob
        max_prob = ec.ept2.max_prob
        step = ec.ept2.step
        used_solvers = ec.ept2.solvers
        res = defaultdict(list)
        for prob in np.arange(min_prob, max_prob, step):
            test_data = WRSNDataset(num_sensors, num_targets, dp.test_size, seed)
            data_loader = DataLoader(test_data, 1, False, num_workers=0)
            wp.k_bit = ec.ept2.k_bit * prob
            for name, solver in solvers.items():
                if name in used_solvers:
                    ret = solver(data_loader, name, save_dir)
                    res[name].append((prob, ret))

        pdump(res, 'data.pickle', save_dir)


    save_dir = os.path.join(save_dir, f'ept_{ept}')
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)
    if not os.path.isfile(os.path.join(save_dir, 'data.pickle')) or rerun:
        if ept == 1:
            run_ept_1(save_dir)
        elif ept == 2:
            run_ept_2(save_dir)

    data = pload('data.pickle', save_dir)
    lifetimes = dict()
    node_failures = dict()
    aggregated_ecr = dict()
    idx = None
    for name, model_data in data.items():
        idx = []
        lifetime_mean = []
        lifetime_std = []
        node_failures_mean = []
        node_failures_std = []
        aggregated_ecr_mean = []
        aggregated_ecr_std = []

        for num_sensors, ret in model_data:
            idx.append(num_sensors)
            lifetime_mean.append(ret['lifetime_mean'])
            lifetime_std.append(ret['lifetime_std'])
            node_failures_mean.append(ret['node_failures_mean'])
            node_failures_std.append(ret['node_failures_std'])
            aggregated_ecr_mean.append(ret['aggregated_ecr_mean'])
            aggregated_ecr_std.append(ret['aggregated_ecr_std'])
        lifetimes[name] = (np.array(lifetime_mean), np.array(lifetime_std))
        node_failures[name] = (np.array(node_failures_mean), np.array(node_failures_std))
        aggregated_ecr[name] = (np.array(aggregated_ecr_mean), np.array(aggregated_ecr_std))
        
    x = np.array(idx)
    xlabel = 'no. sensors' if ept == 1 else 'packet generation prob.'

    plot_mean_std(x, lifetimes, xlabel, 'network lifetime', 'lifetime', save_dir)
    plot_mean_std(x, node_failures, xlabel, 'node failures', 'node_failures', save_dir)
    plot_mean_std(x, aggregated_ecr, xlabel, 'agg. energy consumption rate', 'agg_ecr', save_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', default=123, type=int)
    parser.add_argument('--ept', '-e', dest='epts', 
                        action='append', required=True, type=int)
    parser.add_argument('--config', '-cf', default=None, type=str)
    parser.add_argument('--rerun', action='store_true')

    args = parser.parse_args()

    save_dir = 'results'
    if args.config is not None:
        ec.from_file(args.config)
        basename = os.path.splitext(os.path.basename(args.config))[0]
        save_dir = os.path.join(save_dir, basename)

    wp.from_file(ec.wrsn_config)
    dp.from_file(ec.drl_config)

    for ept in set(args.epts):
        if ept == 1 or ept == 2:
            run_ept_1_2(ept, args.seed, save_dir=save_dir, rerun=args.rerun)

