from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .fall_arm.fall_arm import FallArm
from .fall_arm.fall_arm_config import FallArmCfg, FallArmCfgPPO


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register('fall_arm', FallArm, FallArmCfg(), FallArmCfgPPO())
