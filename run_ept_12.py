import sys
sys.path.insert(0, './model005')

from utils import device, pdump, pload
from utils import WRSNDataset
from utils import WrsnParameters, DrlParameters as dp
from model import MCActor
from environment import WRSNEnv
import utils
from ept_config import EptConfig as ec
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import joblib
import copy
import time

import model005
import model002
import imna
import njnp
import gsa

solvers = {
    "model002": model002.run,
    "model005": model005.run_model005,
    "model005_1": model005.run_model005_1,
    "gsa": gsa.run_gsa,
    "imna": imna.run_imna,
    "njnp": njnp.run_njnp,
    "random": model005.run_random,
}

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_mean_std(x, data, xlabel, ylabel, title, save_dir, plot_std=True,
                  yscale=None, smooth_k=1):
    plt.style.use('seaborn-darkgrid')
    
    fig, ax = plt.subplots()
    for name, (mean, std) in data.items():
        ax.plot(x, smooth(mean, smooth_k), label=name)
        if plot_std:
            ax.fill_between(x, 
                            smooth(np.clip(mean - std, 0.0, np.inf), smooth_k), 
                            smooth(np.clip(mean + std, 0.0, np.inf), smooth_k), 
                            alpha=0.2)

    ax.legend(frameon=True)
    if yscale is not None:
        plt.yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    plt.savefig(os.path.join(save_dir, f'{title}.png'), dpi=400)
    plt.close('all')

def plot_inf_data(x, data, xlabel, ylabel, title, save_dir):
    plt.style.use('seaborn-darkgrid')
    
    fig, ax = plt.subplots()
    i = 0
    for name, model_data in data.items():
        ax.plot(x, model_data, label=name, zorder=10-i)
        i += 1

    ax.legend(frameon=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    plt.savefig(os.path.join(save_dir, f'{title}.png'), dpi=400)
    plt.close('all')
    

def run_ept_1_2(ept, seed=123, save_dir='results', rerun=[], wp_default=WrsnParameters, num_proc=4):
    used_solvers = ec.ept2.solvers if ept == 2 else ec.ept1.solvers

    def solver_wrapper(solver, jobs_desc, *args):
        print("running", jobs_desc)
        start_time = time.time()
        ret = solver(*args)
        print("----> done {}, take: {}, mean_lifetime: {}".format(
                                         jobs_desc,
                                         time.time() - start_time,
                                         ret["lifetime_mean"]))
        print("    > mean_step: {}, mean_rewards: {}, k_bit: {}, E_s: {}, E_mc: {}".format(
                                         ret['step_mean'],
                                         ret['reward_mean'],
                                         ret['k_bit'],
                                         ret['E_s'],
                                         ret['E_mc']))
        return ret

    def run_ept_1(save_dir):
        num_targets = ec.ept1.num_targets
        min_num_sensors = ec.ept1.min_num_sensors
        max_num_sensors = ec.ept1.max_num_sensors
        wp = copy.deepcopy(wp_default)
        wp.k_bit = ec.ept1.k_bit
        max_episode_step = ec.max_episode_step

        res = defaultdict(list)
        jobs_args = []
        jobs_desc = []
        for num_sensors in range(min_num_sensors, max_num_sensors):
            test_data = WRSNDataset(num_sensors, num_targets, ec.ept1.test_size, seed)
            data_loader = DataLoader(test_data, 1, False, num_workers=0)
            for name, solver in solvers.items():
                if name in used_solvers:
                    if not os.path.isfile(os.path.join(save_dir, f'{name}.pickle')) or name in rerun:
                        jobs_args.append((data_loader, name, save_dir, wp, max_episode_step))
                        jobs_desc.append((name, num_sensors))

        rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(solver_wrapper)(
            solvers[jobs_desc[i][0]], jobs_desc[i], *jobs_args[i]) for i in range(len(jobs_args)))

        for i, ret in enumerate(rets):
            res[jobs_desc[i][0]].append((jobs_desc[i][1], ret))

        for key, value in res.items():                
            pdump(value, f'{key}.pickle', save_dir)

    def run_ept_2(save_dir):
        num_targets = ec.ept2.num_targets
        num_sensors = ec.ept2.num_sensors
        min_prob = ec.ept2.min_prob
        max_prob = ec.ept2.max_prob
        step = ec.ept2.step
        max_episode_step = ec.max_episode_step

        jobs_args = []
        jobs_desc = []

        res = defaultdict(list)
        test_data = WRSNDataset(num_sensors, num_targets, ec.ept2.test_size, seed)
        data_loader = DataLoader(test_data, 1, False, num_workers=0)

        for prob in np.arange(min_prob, max_prob, step):
            wp = copy.deepcopy(wp_default)
            wp.k_bit = ec.ept2.k_bit * prob

            for name, solver in solvers.items():
                if name in used_solvers:
                    if not os.path.isfile(os.path.join(save_dir, f'{name}.pickle')) or name in rerun:
                        jobs_args.append((data_loader, name, save_dir, wp, max_episode_step))
                        jobs_desc.append((name, prob))

        rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(solver_wrapper)(
            solvers[jobs_desc[i][0]], jobs_desc[i], *jobs_args[i]) for i in range(len(jobs_args)))

        for i, ret in enumerate(rets):
            res[jobs_desc[i][0]].append((jobs_desc[i][1], ret))

        for key, value in res.items():                
            pdump(value, f'{key}.pickle', save_dir)

    save_dir = os.path.join(save_dir, f'ept_{ept}')
    os.makedirs(save_dir, exist_ok=True)
    
    if ept == 1:
        run_ept_1(save_dir)
    elif ept == 2:
        run_ept_2(save_dir)

    data = {}
    for name in used_solvers:
        data[name] = pload(f'{name}.pickle', save_dir)

    lifetimes = dict()
    node_failures = dict()
    aggregated_ecr = dict()
    inf_data = dict()
    idx = None
    for name, model_data in data.items():
        idx = []
        lifetime_mean = []
        lifetime_std = []
        node_failures_mean = []
        node_failures_std = []
        aggregated_ecr_mean = []
        aggregated_ecr_std = []
        inf_model_data = []

        for num_sensors, ret in model_data:
            idx.append(num_sensors)
            lifetime_mean.append(ret['lifetime_mean'])
            lifetime_std.append(ret['lifetime_std'])
            node_failures_mean.append(ret['node_failures_mean'])
            node_failures_std.append(ret['node_failures_std'])
            aggregated_ecr_mean.append(ret['aggregated_ecr_mean'])
            aggregated_ecr_std.append(ret['aggregated_ecr_std'])
            inf_lifetimes = np.array(ret['inf_lifetimes'])
            num_inf_tests = np.sum(np.isinf(inf_lifetimes))
            inf_model_data.append(num_inf_tests/len(inf_lifetimes))

        lifetimes[name] = (np.array(lifetime_mean), np.array(lifetime_std))
        node_failures[name] = (np.array(node_failures_mean), np.array(node_failures_std))
        aggregated_ecr[name] = (np.array(aggregated_ecr_mean), np.array(aggregated_ecr_std))
        inf_data[name] = inf_model_data

    x = np.array(idx)
    xlabel = 'no. sensors' if ept == 1 else 'packet generation prob.'
        
    plot_mean_std(x, lifetimes, xlabel, 'network lifetime', 'lifetime', 
                  save_dir, yscale=None, smooth_k=4)
    plot_mean_std(x, lifetimes, xlabel, 'network lifetime', 'lifetime_log', 
                  save_dir, yscale='log', smooth_k=4)
    plot_mean_std(x, node_failures, xlabel, 'node failures', 'node_failures', save_dir)
    plot_mean_std(x, aggregated_ecr, xlabel, 'agg. energy consumption rate', 'agg_ecr', save_dir)
    plot_inf_data(x, inf_data, xlabel, 'num. instances', 'inf_data', save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', default=124, type=int)
    parser.add_argument('--ept', '-e', dest='epts', 
                        action='append', required=True, type=int)
    parser.add_argument('--config', '-cf', default=None, type=str)
    parser.add_argument('--rerun', dest='rerun', nargs='*')
    parser.add_argument('--num-proc', '-np', default=4, type=int)

    args = parser.parse_args()

    save_dir = 'results'
    if args.config is not None:
        ec.from_file(args.config)
        basename = os.path.splitext(os.path.basename(args.config))[0]
        save_dir = os.path.join(save_dir, basename)

    WrsnParameters.from_file(ec.wrsn_config)
    wp = WrsnParameters(WrsnParameters.to_dict())

    dp.from_file(ec.drl_config)

    torch.manual_seed(args.seed-1)
    np.random.seed(args.seed-2)

    if args.rerun is None:
        rerun = []
    elif len(args.rerun) == 0:
        rerun = solvers.keys()
    else:
        rerun = args.rerun 
    
    for ept in set(args.epts):
        if ept == 1 or ept == 2:
            run_ept_1_2(ept, args.seed, save_dir=save_dir, rerun=rerun, wp_default=wp, num_proc=args.num_proc)

