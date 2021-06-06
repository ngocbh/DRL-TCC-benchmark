import sys
sys.path.insert(0, './model006')

from utils import device, pdump, pload
from utils import WRSNDataset
from utils import WrsnParameters, DrlParameters as dp
from model import MCActor
import main as model006
import os
import torch
from random_strategy import random_decision_maker
from main import decision_maker

def run_model006_2(data_loader, name, save_dir, wp, max_step=1000):
    actor = MCActor(dp.MC_INPUT_SIZE,
                    dp.DEPOT_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)
    save_dir = os.path.join(save_dir, name)
    checkpoint = 'model006/checkpoints/mc_20_10_6_small_2/19'
    path = os.path.join(checkpoint, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))

    ret = model006.validate(data_loader, decision_maker, (actor,), wp=wp, max_step=max_step,
                            render=False, verbose=False)
    return ret


def run_model006_1(data_loader, name, save_dir, wp, max_step=1000):
    actor = MCActor(dp.MC_INPUT_SIZE,
                    dp.DEPOT_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)
    save_dir = os.path.join(save_dir, name)
    checkpoint = 'model006/checkpoints/mc_20_10_6_small_1/19'
    path = os.path.join(checkpoint, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))

    ret = model006.validate(data_loader, decision_maker, (actor,), wp=wp, max_step=max_step,
                            render=False, verbose=False)
    return ret

def run_random(data_loader, name, save_dir, wp, max_step=1000):
    save_dir = os.path.join(save_dir, name)
    return model006.validate(data_loader, random_decision_maker, wp=wp, normalize=False, max_step=max_step)
