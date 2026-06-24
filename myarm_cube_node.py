import numpy as np
from scipy.stats import qmc
from .SE3_utils import hat, exp, Log
from .parameters import screws, Tws, Tsb, q_lb, q_ub


def poe_fk(q, frame_in = "space", frame_ref = "world"):
    q = np.minimum(np.maximum(q, q_lb), q_ub)
    poe = Tsb
    if frame_in == "space":
        for k in range(len(q) - 1, -1, -1):
            poe = exp(hat(screws[frame_in][k]) * q[k]) @ poe
    elif frame_in == "body":
        for k in range(0, len(q), 1):
            poe = poe @ exp(hat(screws[frame_in][k]) * q[k])
    if frame_ref == "world":
        return Tws @ poe
    elif frame_ref == "space":
        return poe
    elif frame_ref == "body":
        return np.eye(4)
    return poe  # default is the world frame

# find configurations q closed to a desired one qd, searching for poses T close to a desired one Td
# choose random initial configurations using Sobol sequence
def find_q_close_to_qd(Td, frame_in = "space", frame_ref = "world", sample_size = 2**8):
    sampler = qmc.Sobol(d = len(q_lb), scramble = True)
    unit_samples = sampler.random(sample_size)
    q_samples = qmc.scale(unit_samples, q_lb, q_ub)
    errors = np.zeros(sample_size)
    for k in range(sample_size):
        T = poe_fk(q_samples[k], frame_in, frame_ref)
        errors[k] = np.linalg.norm(Log(Td @ np.linalg.inv(T)))
        # errors[k] = np.linalg.norm(Td[:3, 3] - T[:3, 3])
    sorted_indices = np.argsort(errors)
    sorted_errors = errors[sorted_indices]
    sorted_q_samples = q_samples[sorted_indices]
    return sorted_q_samples, sorted_errors
