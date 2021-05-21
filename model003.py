import sys
from model003.model import MCActor as MCActor003

import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

sys.path.append('./model004')
from utils import device, pdump, pload
from utils import WRSNDataset
from utils import WrsnParameters as wp, DrlParameters as dp
import os
import torch
import random_strategy
import main as model002
import utils

print(utils.__file__)
print(model002.__file__)
print(wp.p_request_threshold)

def run_model003(data_loader, name, save_dir, wp, max_step=1000):
    actor = MCActor003(dp.MC_INPUT_SIZE,
                    dp.DEPOT_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)

    save_dir = os.path.join(save_dir, name)
    checkpoint = 'model003/checkpoints/mc_20_10_3_small/0'
    path = os.path.join(checkpoint, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))

    ret = model002.validate(data_loader, model002.decision_maker, (actor,), wp=wp, max_step=max_step,
                            render=False, verbose=False)
    return ret

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)
    seed=123
    torch.manual_seed(seed-1)
    np.random.seed(seed-2)
    dataset = WRSNDataset(20, 10, 1, 1)
    wp.k_bit = 6000000
    data_loader = DataLoader(dataset, 1, False, num_workers=0)
    actor = MCActor003(dp.MC_INPUT_SIZE,
                    dp.DEPOT_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)

    checkpoint = 'model003/checkpoints/mc_20_10_3_small/4'
    path = os.path.join(checkpoint, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))

    model002.validate(data_loader, model002.decision_maker, (actor,), wp=wp, max_step=1000,
                            render=False, verbose=False)