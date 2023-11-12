import isaacgym

import os
import hydra
import datetime
from termcolor import cprint
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from handem.tasks import task_map
from handem.utils.reformat import omegaconf_to_dict  # , print_dict
from handem.utils.utils import set_np_formatting, set_seed  # , git_hash, git_diff_config


## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg == "" else arg)


@hydra.main(config_name="config", config_path="./config")
def main(cfg: DictConfig):
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed)

    cprint("Start Building the Environment", "green", attrs=["bold"])
    env = task_map[cfg.task_name](
        cfg=omegaconf_to_dict(cfg.task),
        rl_device=cfg.rl_device,
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=False,
        force_render=False,
    )

    env.reset()
    while True:
        actions = env.random_actions()
        _ = env.step(actions)
        if not cfg.headless:
            env.render()


if __name__ == "__main__":
    main()
