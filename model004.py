import sys
sys.path.insert(0, './model002')

from utils import device, pdump, pload
from utils import WRSNDataset
from utils import WrsnParameters as wp, DrlParameters as dp
from model import MCActor
import main as model002
import os
import torch
from random_strategy import random_decision_maker
from main import decision_maker

def run_model004(data_loader, name, save_dir, wp, max_step=1000):
    actor = MCActor(dp.MC_INPUT_SIZE,
                    dp.DEPOT_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)

    save_dir = os.path.join(save_dir, name)
    checkpoint = 'model002/checkpoints/mc_20_10_2_small_1/6'
    path = os.path.join(checkpoint, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))

    ret = model002.validate(data_loader, decision_maker, (actor,), wp=wp, max_step=max_step,
                            render=False, verbose=False)
    return ret