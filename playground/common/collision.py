"""Vendored from mujoco_playground._src.collision (removed upstream in playground 0.2.0)."""
import jax
import jax.numpy as jp
from mujoco import mjx


def get_collision_info(contact, geom1: int, geom2: int):
    mask = (jp.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (jp.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = jp.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    normal = (dist < 0) * contact.frame[idx, 0, :3]
    return dist, normal


def geoms_colliding(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
    return get_collision_info(state.contact, geom1, geom2)[0] < 0
