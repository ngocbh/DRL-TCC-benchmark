import sys
sys.path.insert(0, './model002')

from utils import device, pdump, pload
from utils import WRSNDataset
from utils import WrsnParameters, DrlParameters
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
from main import decision_maker
from random_strategy import random_decision_maker
from imna import imna_decision_maker
import model002
import imna


def validate(data_loader, decision_maker, args=None, wp=WrsnParameters,
             render=False, verbose=False, max_step=None, normalize=True):

    rewards = []
    mean_policy_losses = []
    mean_entropies = []
    times = [0]
    net_lifetimes = []
    mc_travel_dists = []
    inf_lifetimes = []

    for idx, data in enumerate(data_loader):
        if verbose: print("Test %d" % idx)

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
            # mask[0] = 1.0
                
            mc_state = torch.from_numpy(mc_state).to(dtype=torch.float32, device=device)
            depot_state = torch.from_numpy(depot_state).to(dtype=torch.float32, device=device)
            sn_state = torch.from_numpy(sn_state).to(dtype=torch.float32, device=device)

            if verbose: 
                print("Step %d: Go to %d (prob: %2.4f) => reward (%2.4f, %2.4f)\n" % 
                      (step, action, prob, reward[0], reward[1]))
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

        inf_lifetimes.append(env.get_network_lifetime() 
                             if done else np.inf)

    ret = {}
    ret['inf_lifetimes'] = inf_lifetimes
    return ret

def run(data_loader, name, save_dir, max_step=1000):
    actor = MCActor(dp.MC_INPUT_SIZE,
                    dp.DEPOT_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)

    save_dir = os.path.join(save_dir, name)
    checkpoint = 'model002/checkpoints/mc_20_10_2_small/6'
    path = os.path.join(checkpoint, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))

    ret = validate(data_loader, decision_maker, (actor,), max_step=max_step,
                            render=False, verbose=False)
    return ret

def run_random(data_loader, name, save_dir, max_step=1000):
    save_dir = os.path.join(save_dir, name)
    return validate(data_loader, random_decision_maker, normalize=False, max_step=max_step)


solvers = {
    "model002": run_model002,
    "imna": run_imna,
    "random": run_random,
}

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def run_ept_3(seed=123, save_dir='results', rerun=[]):
    used_solvers = ec.ept3.solvers

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

        for prob in np.arange(min_prob, max_prob, step):
            wp = WrsnParameters()
            wp.k_bit = ec.ept3.k_bit * prob
            print(wp.k_bit)
            for name, solver in solvers.items():
                if name in used_solvers:
                    if not os.path.isfile(os.path.join(save_dir, f'{name}.pickle')) or name in rerun:
                        print(f"running on {prob, name}")
                        ret = solver(data_loader, name, save_dir, max_episode_step)
                        res[name].append((prob, ret))
            
        for key, value in res.items():                
            pdump(value, f'{key}.pickle', save_dir)

    def plot(x, data, xlabel, ylabel, title, save_dir, plot_std=True,
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

    save_dir = os.path.join(save_dir, f'ept_{3}')
    os.makedirs(save_dir, exist_ok=True)

    run(save_dir)

    data = {}
    for name in used_solvers:
        data[name] = pload(f'{name}.pickle', save_dir)

    lifetimes = dict()
    idx = None
    for name, model_data in data.items():
        idx = []
        inf_lifetimes = []

        for _id, ret in model_data:
            idx.append(_id)
            inf_lifetimes.append(ret['inf_lifetimes'])
        lifetimes[name] = (np.array(inf_lifetimes), np.array(lifetime_std))

    x = np.array(idx)
    xlabel = 'packet generation prob.'
    plot(data)


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
    DrlParameters.from_file(ec.drl_config)

    torch.manual_seed(args.seed-1)
    np.random.seed(args.seed-2)

    if args.rerun is None:
        rerun = []
    elif len(args.rerun) == 0:
        rerun = solvers.keys()
    else:
        rerun = args.rerun 

    run_ept_3(args.seed, save_dir=save_dir, rerun=rerun)