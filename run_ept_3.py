import sys
sys.path.insert(0, './model002')

from utils import device, pdump, pload
from utils import WRSNDataset
from utils import WrsnParameters, DrlParameters as dp
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
import joblib
import time
from main import decision_maker
from random_strategy import random_decision_maker
from imna import imna_decision_maker
from ept_config import EptConfig
import model002
import imna
import njnp
import itertools
import gsa



def validate(data_loader, decision_maker, args=None, 
             wp=WrsnParameters, prob_range=(0.3, 0.4), max_step=None,
             render=False, verbose=False, normalize=True,
             on_validation_begin=None, on_validation_end=None, 
             on_episode_begin=None, on_episode_end=None):
    start_time = time.time()

    if on_validation_begin is not None:
        on_validation_begin(*args)

    rewards = []
    inf_lifetimes = []
    k_bit = wp.k_bit

    for idx, data in enumerate(data_loader):
        package_generation_prob = np.random.uniform(*prob_range)
        wp.k_bit = k_bit * package_generation_prob
        # print("Test {}, {}, {}".format(idx, wp.k_bit, decision_maker.__name__))

        sensors, targets = data
        
        env = WRSNEnv(sensors=sensors.squeeze(), 
                      targets=targets.squeeze(), 
                      wp=wp,
                      normalize=normalize)

        mc_state, depot_state, sn_state = env.reset()
        
        mc_state = torch.from_numpy(mc_state).to(dtype=torch.float32, device=device)
        depot_state = torch.from_numpy(depot_state).to(dtype=torch.float32, device=device)
        sn_state = torch.from_numpy(sn_state).to(dtype=torch.float32, device=device)

        rewards = []

        mask = torch.ones(env.action_space.n).to(device)

        if on_episode_begin is not None:
            on_episode_begin(*args)
        # verbose = True
        # render=True
        max_step = max_step or dp.max_step
        for step in range(max_step):
            if render:
                env.render()

            if args is not None:
                action, prob = decision_maker(mc_state, depot_state, sn_state, mask, *args)
            else:
                action, prob = decision_maker(mc_state, depot_state, sn_state, mask)
            
            mask[env.last_action] = 1.0
            (mc_state, depot_state, sn_state), reward, done, _ = env.step(action)
            mask[env.last_action] = 0.0
                
            mc_state = torch.from_numpy(mc_state).to(dtype=torch.float32, device=device)
            depot_state = torch.from_numpy(depot_state).to(dtype=torch.float32, device=device)
            sn_state = torch.from_numpy(sn_state).to(dtype=torch.float32, device=device)

            if verbose: 
                print("Step %d: Go to %d (prob: %2.4f) => reward (%2.4f, %2.4f)\n" % 
                      (step, action, prob, reward[0], reward[1]))
                print("Aggregated ecr %2.4f, node failures %2.4f\n" % 
                       (env.net.aggregated_ecr, env.net.node_failures))
                print("Current network lifetime: %2.4f, mc_battery: %2.4f \n\n" % 
                       (env.net.network_lifetime, env.mc.cur_energy))

            rewards.append(reward)

            if done:
                if verbose: print("End episode! Press any button to continue...")
                if render:
                    env.render()
                    input()
                env.close()
                break

            if render:
                time.sleep(0.5)
                # pass
        if on_episode_end is not None:
            on_episode_end(*args)

        lifetime = env.get_network_lifetime() if done else np.inf
        inf_lifetimes.append((package_generation_prob, lifetime))

    wp.k_bit = k_bit
    ret = {}
    ret['inf_lifetimes'] = inf_lifetimes
    
    if on_validation_end is not None:
        on_validation_end(*args)

    # print("Validation time : {}".format(time.time() - start_time))
    return ret

def run_model002(data_loader, name, save_dir, wp, prob_range, max_step=1000):
    actor = MCActor(dp.MC_INPUT_SIZE,
                    dp.DEPOT_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)

    save_dir = os.path.join(save_dir, name)
    checkpoint = 'model002/checkpoints/mc_20_10_2_small/6'
    path = os.path.join(checkpoint, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))

    ret = validate(data_loader, decision_maker, (actor,), wp=wp, 
                   prob_range=prob_range, max_step=max_step, normalize=True)
    return ret

def run_random(data_loader, name, save_dir, wp, prob_range, max_step=1000):
    save_dir = os.path.join(save_dir, name)
    return validate(data_loader, random_decision_maker, wp=wp, 
                    prob_range=prob_range, max_step=max_step, normalize=False)


def run_imna(data_loader, name, save_dir, wp, prob_range, max_step=1000):
    return validate(data_loader, imna.imna_decision_maker, wp=wp, 
                    prob_range=prob_range, max_step=max_step, normalize=False)

def run_njnp(data_loader, name, save_dir, wp, prob_range, max_step=1000):
    return validate(data_loader, njnp.njnp_decision_maker, wp=wp, 
                    prob_range=prob_range, max_step=max_step, normalize=False)

def run_gsa(data_loader, name, save_dir, wp, prob_range, max_step=1000):
    def gsa_reset(gsa):
        gsa.reset()

    gsa_ = gsa.GSA()
    return validate(data_loader, gsa.gsa_decision_maker, (gsa_,), wp=wp, prob_range=prob_range, 
                    normalize=False, max_step=max_step, on_episode_begin=gsa_reset)


solvers = {
    "model002": run_model002,
    "gsa": run_gsa,
    "imna": run_imna,
    "njnp": run_njnp,
    "random": run_random,
}

label_map = {
    "model002": "model002",
    "gsa": "gsa",
    "imna": "imna",
    "njnp": "njnp",
    "random": "random",
}

def run_ept_3(seed=123, save_dir='results', rerun=[]):
    used_solvers = ec.ept3.solvers

    def solver_wrapper(solver, jobs_desc, *args):
        print("running", jobs_desc)
        start_time = time.time()
        ret = solver(*args)
        print("done {}, take: {}".format(jobs_desc, 
                                         time.time() - start_time))
        return ret

    def run(save_dir):
        num_targets = ec.ept3.num_targets
        num_sensors = ec.ept3.num_sensors
        min_prob = ec.ept3.min_prob
        max_prob = ec.ept3.max_prob
        step = ec.ept3.step
        max_episode_step = ec.max_episode_step

        res = defaultdict(list)
        test_data = WRSNDataset(num_sensors, num_targets, ec.ept3.test_size, seed)
        data_loader = DataLoader(test_data, 1, False, num_workers=0)

        jobs_args = []
        jobs_desc = []

        for prob in np.arange(min_prob, max_prob, step):
            wp = WrsnParameters()
            wp.k_bit = ec.ept3.k_bit
            prob_range = (prob, prob + step)

            for name, solver in solvers.items():
                if name in used_solvers:
                    if not os.path.isfile(os.path.join(save_dir, f'{name}.pickle')) or name in rerun:
                        jobs_args.append((data_loader, name, save_dir, wp, prob_range, max_episode_step))
                        jobs_desc.append((name, prob_range))

        rets = joblib.Parallel(n_jobs=8)(joblib.delayed(solver_wrapper)(
            solvers[jobs_desc[i][0]], jobs_desc[i], *jobs_args[i]) for i in range(len(jobs_args)))

        for i, ret in enumerate(rets):
            res[jobs_desc[i][0]].append((jobs_desc[i][1], ret))
            
        for key, value in res.items():                
            pdump(value, f'{key}.pickle', save_dir)

    def plot(data, save_dir):
        # plt.style.use('seaborn-white')

        fig, ax = plt.subplots()
        lifetime_values = []
        for e in data.values():
            lifetime_values.extend(e[1])
        lifetime_values = np.array(lifetime_values)
        finite = np.isfinite(lifetime_values)
        max_value = np.max(lifetime_values[finite])

        max_scale = 1.8
        step = 0.22
        margin = 0.2
        i = 0
        marker = itertools.cycle(['x', '*', 'v', '^', "s", "v", "^"])
        for name, (probs, lifetimes) in data.items():
            print(name)
            lifetimes = np.array(lifetimes)
            inf_idx = np.isinf(lifetimes)
            lifetimes[inf_idx] = max_value * (max_scale - i *step)
            ax.scatter(probs, lifetimes, label=label_map[name], alpha=0.5, s=10, 
                        marker=next(marker), plotnonfinite=True, zorder=10-i)
            i += 1

        ax.axhline(y=max_value * (max_scale-(i-1)*step-margin), color="black", linestyle=":", linewidth=1)
        ax.legend(frameon=True)
        plt.yscale('log')
       

        ax.set_ylim(top=max_value*(max_scale + margin))

        ax.set_xlabel('packet generation prob.')
        ax.set_ylabel('lifetimes')
        plt.text(0.24, .96, 'INF',
                transform=ax.get_xaxis_transform(),
                horizontalalignment='center',
                weight=12, color='black',
                fontdict={'fontfamily': 'monospace'})
        plt.savefig(os.path.join(save_dir, 'ept3.png'), dpi=400)
        plt.close('all')

    save_dir = os.path.join(save_dir, f'ept_{3}')
    os.makedirs(save_dir, exist_ok=True)

    run(save_dir)

    data = {}
    for name in used_solvers:
        data[name] = pload(f'{name}.pickle', save_dir)

    normalized_data = dict()
    idx = None
    for name, model_data in data.items():
        idx = []
        inf_lifetimes = []
        x = []
        for ft, e in model_data:
            x.extend(e['inf_lifetimes'][:300])
        for prob, lifetime in x:
            idx.append(prob)
            inf_lifetimes.append(lifetime)
        normalized_data[name] = (idx, inf_lifetimes)
        # print(name, list(zip(idx, inf_lifetimes))  ) 

    plot(normalized_data, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', default=123, type=int)
    parser.add_argument('--config', '-cf', default=None, type=str)
    parser.add_argument('--rerun', dest='rerun', nargs='*')

    args = parser.parse_args()

    save_dir = 'results'
    if args.config is not None:
        ec.from_file(args.config)
        basename = os.path.splitext(os.path.basename(args.config))[0]
        save_dir = os.path.join(save_dir, basename)

    WrsnParameters.from_file(ec.wrsn_config)
    dp.from_file(ec.drl_config)

    torch.manual_seed(args.seed-1)
    np.random.seed(args.seed-2)

    if args.rerun is None:
        rerun = []
    elif len(args.rerun) == 0:
        rerun = solvers.keys()
    else:
        rerun = args.rerun 

    run_ept_3(args.seed, save_dir=save_dir, rerun=rerun)
