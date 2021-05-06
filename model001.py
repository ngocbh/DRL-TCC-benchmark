import sys
sys.path.insert(0, './model001')
from model import MCActor as MCActor001
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
sys.path.insert(0, './model002')
from utils import device, pdump, pload
from utils import WRSNDataset
from utils import WrsnParameters as wp, DrlParameters as dp
import os
import torch
import random_strategy
import main as model002

def model001_decision_maker(mc_state, depot_state, sn_state, mask, actor):
    actor.eval()
    mc_state = mc_state.unsqueeze(0)
    depot_state = depot_state.unsqueeze(0)
    sn_state = sn_state.unsqueeze(0)

    with torch.no_grad():
        logit = actor(mc_state, sn_state)

    logit = logit + mask[1:].log()
    prob = F.softmax(logit, dim=-1)

    prob, action = torch.max(prob, 1)  # Greedy selection
    actor.train()
    return action.squeeze().item(), prob

def run_model001(data_loader, name, save_dir, wp, max_step=1000):
    actor = MCActor001(dp.MC_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)

    save_dir = os.path.join(save_dir, name)
    checkpoint = 'model002/checkpoints/mc_20_10_0/21'
    path = os.path.join(checkpoint, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))

    ret = model002.validate(data_loader, model001_decision_maker, (actor,), wp=wp, max_step=max_step,
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
    actor = MCActor001(dp.MC_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)

    checkpoint = 'model002/checkpoints/mc_20_10_0/21'
    path = os.path.join(checkpoint, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))
    max_step = 1000

    ret = model002.validate(data_loader, model001_decision_maker, (actor,), wp=wp, max_step=max_step,
                            render=False, verbose=True)