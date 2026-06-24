"""
pd_controller.py
----------------
PD Controller για το Go2.

Ο νόμος ελέγχου είναι απλός:
    torque = Kp * (q_target - q) - Kd * q_dot

    q       = τρέχουσες γωνίες joints
    q_dot   = τρέχουσες ταχύτητες joints
    q_target= η standing pose (default θέση)
    Kp      = πόσο δυνατά διορθώνει τη θέση
    Kd      = πόσο αντιστέκεται στην ταχύτητα (damping)

Δεν μαθαίνει τίποτα — απλοί κανόνες.
Στόχος: να κρατάει τον Go2 όρθιο.
"""

import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv

# Standing pose — μετρήθηκε με get_default_pose.py
STANDING_POSE = np.array([
     0.21291979,  0.67369252, -1.64459852,  # FL
    -0.20640879,  0.84122732, -1.94411832,  # FR
     0.29783284,  0.94530330, -1.69789124,  # RL
     0.12787365,  1.01380836, -1.57726899,  # RR
])

OBS = (
    "gravity_vector:base",
    "base_lin_acc:base",
    "base_ang_vel:base",
    "qpos_js",
    "qvel_js",
    "base_lin_vel_err:base",
    "base_ang_vel_err:base",
)


class PDController:
    def __init__(self, kp=40.0, kd=1.0):
        self.kp = kp
        self.kd = kd
        self.target = STANDING_POSE.copy()

    def reset(self):
        pass  # ο PD δεν έχει εσωτερική κατάσταση

    def act(self, obs):
        q     = obs["qpos_js"]   # τρέχουσες γωνίες
        q_dot = obs["qvel_js"]   # τρέχουσες ταχύτητες

        # PD νόμος
        torques = self.kp * (self.target - q) - self.kd * q_dot

        return torques.astype(np.float32)


if __name__ == "__main__":
    env = QuadrupedEnv(
        robot="go2",
        scene="flat",
        base_vel_command_type="random",
        state_obs_names=OBS,
    )

    controller = PDController(kp=40.0, kd=1.0)

    print("PD Controller — 3 episodes\n")

    for ep in range(3):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        controller.reset()
        steps = 0

        for t in range(3000):
            action = controller.act(obs)
            result = env.step(action)
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            steps += 1
            env.render()
            if done:
                break

        print(f"  Episode {ep+1}: {steps} steps πριν πέσει")

    print(f"\nΣύγκριση:")
    print(f"  Random policy : ~100 steps")
    print(f"  PD Controller : ??? steps  <- αυτό μόλις μετρήσαμε")

    env.close()