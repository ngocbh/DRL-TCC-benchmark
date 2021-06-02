import sys
sys.path.insert(0, './model005')

from utils import device, pdump, pload
from utils import WRSNDataset
from utils import WrsnParameters, DrlParameters as dp
from model import MCActor
import main as model005
import os
import torch
from random_strategy import random_decision_maker
from main import decision_maker

def run_model005(data_loader, name, save_dir, wp, max_step=1000):
    actor = MCActor(dp.MC_INPUT_SIZE,
                    dp.DEPOT_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)
    save_dir = os.path.join(save_dir, name)
    checkpoint = 'model005/checkpoints/mc_20_10_5_small/6'
    path = os.path.join(checkpoint, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))

    ret = model005.validate(data_loader, decision_maker, (actor,), wp=wp, max_step=max_step,
                            render=False, verbose=False)
    return ret


def run_model005_1(data_loader, name, save_dir, wp, max_step=1000):
    actor = MCActor(dp.MC_INPUT_SIZE,
                    dp.DEPOT_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)
    save_dir = os.path.join(save_dir, name)
    checkpoint = 'model005/checkpoints/mc_20_10_5_small_1/10'
    path = os.path.join(checkpoint, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))

    ret = model005.validate(data_loader, decision_maker, (actor,), wp=wp, max_step=max_step,
                            render=False, verbose=False)
    return ret

def run_random(data_loader, name, save_dir, wp, max_step=1000):
    save_dir = os.path.join(save_dir, name)
    return model005.validate(data_loader, random_decision_maker, wp=wp, normalize=False, max_step=max_step)
