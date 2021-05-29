import sys
sys.path.append('./model005')

from utils import Config
import argparse


class Ept1(Config):
    __dictpath__ = 'ec.ept1'

    # ept 1 settings
    solvers = ['model002', 'gsa', 'imna', 'njnp', 'random']
    num_targets = 10
    test_size = 10
    min_num_sensors = 20
    max_num_sensors = 30
    k_bit = 8000000

class Ept2(Config):
    __dictpath__ = 'ec.ept2'

    solvers = ['model002', 'gsa', 'imna', 'njnp', 'random']
    num_targets = 10
    num_sensors = 20
    test_size = 10
    k_bit = 20000000
    step = 0.1
    min_prob = 0.1
    max_prob = 1

class Ept3(Config):
    __dictpath__ = 'ec.ept3'

    solvers = ['model002', 'gsa', 'imna', 'njnp', 'random']
    num_targets = 10
    num_sensors = 20
    test_size = 10
    k_bit = 20000000
    step = 0.1
    min_prob = 0.3
    max_prob = 1
    repeat = 1
    

class Ept4(Config):
    __dictpath__ = 'ec.ept4'

    # ept 1 settings
    solvers = ['model002', 'gsa', 'imna', 'njnp', 'random']
    num_sensors = 20
    test_size = 10
    min_num_targets = 10
    max_num_targets = 20
    k_bit = 24000
    


class EptConfig(Config):
    __dictpath__ = 'ec'

    # other configs
    wrsn_config = 'model002/configs/dev.yml'
    drl_config = 'model002/configs/dev.yml'
    checkpoint = 'model002/checkpoints/mc_20_10_0/21'
    max_episode_step = 2000

    ept1 = Ept1
    ept2 = Ept2
    ept3 = Ept3
    ept4 = Ept4



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--dump', default='config.yml', type=str)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--mode', default='merge_cls', type=str)

    args = parser.parse_args()
    if args.load is not None:
        EptConfig.from_file(args.load)
    EptConfig.to_file(args.dump, mode=args.mode)
