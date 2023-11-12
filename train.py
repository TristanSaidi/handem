# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

import isaacgym

import os
import hydra
import datetime
from termcolor import cprint
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from handem.algo.ppo.ppo import PPO
from handem.tasks import task_map
from handem.utils.reformat import omegaconf_to_dict, print_dict
from handem.utils.misc import set_np_formatting, set_seed, git_hash, git_diff_config


## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config.
# used primarily for num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


@hydra.main(config_name='config', config_path='config', version_base="1.2")
def main(cfg: DictConfig):
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    env = task_map[cfg.task_name](
        cfg=omegaconf_to_dict(cfg.task),
        rl_device=cfg.rl_device,
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=False,
        force_render=False,
    )

    output_dif = os.path.join('outputs', cfg.train.ppo.output_name)
    os.makedirs(output_dif, exist_ok=True)
    agent = eval(cfg.train.algo)(env, output_dif, full_config=cfg)
    if cfg.test:
        agent.restore_test(cfg.train.load_path)
        agent.test()
    else:
        date = str(datetime.datetime.now().strftime('%m%d%H'))
        with open(os.path.join(output_dif, f'config_{date}.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

        # check whether execute train by mistake:
        best_ckpt_path = os.path.join(
            'outputs', cfg.train.ppo.output_name, "nn"
        )
        if os.path.exists(best_ckpt_path):
            user_input = input(
                f'are you intentionally going to overwrite files in {cfg.train.ppo.output_name}, type yes to continue \n')
            if user_input != 'yes':
                exit()

        agent.restore_train(cfg.train.load_path)
        agent.train()


if __name__ == '__main__':
    main()
