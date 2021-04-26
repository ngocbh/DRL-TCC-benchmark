import sys
sys.path.append('./drl')

from utils import Config
import argparse


class Ept1(Config):
    __dictpath__ = 'ec.ept1'

    # ept 1 settings
    solvers = ['drl', 'random']
    num_targets = 10
    test_size = 10
    min_num_sensors = 20
    max_num_sensors = 30

class Ept2(Config):
    __dictpath__ = 'ec.ept2'

    solvers = ['drl', 'random']
    num_targets = 10
    num_sensors = 20
    test_size = 10
    k_bit = 20000000
    step = 0.1
    min_prob = 0.1
    max_prob = 1


class EptConfig(Config):
    __dictpath__ = 'ec'

    # other configs
    wrsn_config = 'drl/configs/dev.yml'
    drl_config = 'drl/configs/dev.yml'

    checkpoint = 'drl/checkpoints/mc_20_10_0/21'

    ept1 = Ept1
    ept2 = Ept2



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--dump', default='config.yml', type=str)
    parser.add_argument('--mode', default='merge_cls', type=str)

    args = parser.parse_args()
    EptConfig.to_file(args.dump, mode=args.mode)
