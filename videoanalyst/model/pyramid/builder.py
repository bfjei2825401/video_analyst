# -*- coding: utf-8 -*
from typing import Dict, List

from loguru import logger
from yacs.config import CfgNode

from videoanalyst.utils import merge_cfg_into_hps

from .pyramid_base import TASK_PYRAMIDS, TRACK_PYRAMIDS, VOS_PYRAMIDS


def build(task: str, cfg: CfgNode, pyramid_model=None):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        builder configuration

    pyramid_model:
        warp pyramid into encoder if not None

    Returns
    -------
    torch.nn.Module
        module built by builder
    """
    if task in TASK_PYRAMIDS:
        modules = TASK_PYRAMIDS[task]
    else:
        logger.error("no pyramid for task {}".format(task))
        exit(-1)

    name = cfg.name
    assert name in modules, "pyramid {} not registered for {}!".format(
        name, task)

    if pyramid_model:
        module = modules[name](pyramid_model)
    else:
        module = modules[name]()

    hps = module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    module.set_hps(hps)
    module.update_params()
    return module


def get_config(task_list: List) -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {task: CfgNode() for task in task_list}
    for cfg_name, module in TASK_PYRAMIDS.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in module:
            cfg[name] = CfgNode()
            pyramid = module[name]
            hps = pyramid.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
