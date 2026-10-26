"""Shims for mujoco_playground API changes (0.0.x -> 0.2.0)."""
from mujoco import mjx
from mujoco_playground._src import mjx_env


def init(model, qpos=None, qvel=None, ctrl=None):
    """Old mjx_env.init: make data, set state, run forward kinematics."""
    data = mjx_env.make_data(model, qpos=qpos, qvel=qvel, ctrl=ctrl)
    return mjx.forward(model, data)
