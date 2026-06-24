import numpy as np
from .poe_fkine import poe_fk, find_q_close_to_qd
from .poe_ikine import poe_ik
from .SE3_utils import Log
from .parameters import N, Tws


def solve_ik(Twbd, frame_ref, q0_insert = None, error_tol = 1e-5, max_ikine_iter = 100, max_q0_search = 2**6, max_q0_tries = 2**4, compute_frame = "space", avoid_box=False):
    # find initial configurations q closed to the desired one qd, searching for poses Twb close to the desired one Twbd
    # choose random initial configurations using Sobol sequence
    # then, apply inverse kinematics to the best initial configurations (the ones closer to the solution)

    if frame_ref == "world" or frame_ref == "body":
        Td = np.copy(Twbd)
    elif frame_ref == "space":
        Td = np.linalg.inv(Tws) @ Twbd

    qd = np.zeros(N)
    success_ikine = False
    if q0_insert is None:
        q0_list, dist_list = find_q_close_to_qd(Td, compute_frame, "world" if frame_ref == "body" else frame_ref, max_q0_search)
    else:
        q0_list = list(q0_insert)
        max_q0_tries = len(q0_list)
    for j in range(max_q0_tries):
        try:
            q0 = np.copy(q0_list[j])
        except:
            break
        qd = poe_ik(q0, Td, compute_frame, frame_ref, max_ikine_iter, error_tol, avoid_box=avoid_box)
        Td_verify = poe_fk(qd, compute_frame, "world" if frame_ref == "body" else frame_ref)
        if np.linalg.norm(Log(Td @ np.linalg.inv(Td_verify))) < error_tol:
            success_ikine = True
            break

    return qd, success_ikine
