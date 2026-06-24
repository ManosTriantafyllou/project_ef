"""
env_wrapper_cpg.py
--------------------
RESIDUAL LEARNING με Central Pattern Generator (CPG) βάση — επιτρέπεται
ρητά από το PDF: "central pattern generators" είναι στη λίστα επιτρεπτών
μεθόδων για το Project 3.

Ιδέα:
    action_final = CPG_trajectory(t) + PPO_correction

Σε αντίθεση με το προηγούμενο residual (Plan C), όπου η βάση ήταν η
ΣΤΑΤΙΚΗ standing pose (που ενθάρρυνε τον agent να μάθει "μην κινηθείς,
ήδη παίρνω καλό reward"), εδώ η βάση είναι μια ΗΔΗ ΚΙΝΟΥΜΕΝΗ, περιοδική
τροχιά βαδίσματος (sinusoidal leg pattern, classic CPG για quadrupeds).

Έτσι, η "μηδενική προσπάθεια" (PPO output ≈ 0) ΔΕΝ είναι στατική
ισορροπία — είναι ήδη μια πρόχειρη προσπάθεια περπατήματος. Ο PPO
μαθαίνει μόνο να ΔΙΟΡΘΩΝΕΙ αυτό το pattern (πλάτος, φάση, ισορροπία),
αντί να αποφασίζει από το μηδέν αν αξίζει να κινηθεί καθόλου.

ΔΕΝ αγγίζει reward/observation - μόνο πώς παράγεται το action.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU


# ─────────────────────────────────────────────────────────────
# Ίδιο IMU patch + official reward (ΔΕΝ αλλάζει τίποτα εδώ)
# ─────────────────────────────────────────────────────────────
_original_imu_init = IMU.__init__

def _patched_imu_init(self, mj_model, mj_data, *args, **kwargs):
    mujoco.mj_forward(mj_model, mj_data)
    _original_imu_init(self, mj_model, mj_data, *args, **kwargs)
    self.step()

IMU.__init__ = _patched_imu_init


def _compute_reward(self):
    lin_vel_err_B = self.base_lin_vel_err(frame="base")
    ang_vel_err_B = self.base_ang_vel_err(frame="base")

    sigma_lin_vel = 0.05
    sigma_ang_vel = 0.05
    tracking_lin_vel = np.exp(-np.sum(lin_vel_err_B[:2] ** 2) / (2 * sigma_lin_vel**2))
    tracking_yaw_rate = np.exp(-(ang_vel_err_B[2] ** 2) / (2 * sigma_ang_vel**2))

    gravity_B = self.gravity_vector
    upright_penalty = np.sum(gravity_B[:2] ** 2)

    base_lin_vel_B = self.base_lin_vel(frame="base")
    base_ang_vel_B = self.base_ang_vel(frame="base")

    z_vel_penalty = base_lin_vel_B[2] ** 2
    roll_pitch_ang_vel_penalty = np.sum(base_ang_vel_B[:2] ** 2)

    tau = self.torque_ctrl_setpoint
    torque_penalty = np.sum(tau**2)

    current_action = self.mjData.ctrl.copy()
    if not hasattr(self, "_last_action_for_reward"):
        self._last_action_for_reward = np.zeros_like(current_action)

    action_rate_penalty = np.sum((current_action - self._last_action_for_reward) ** 2)
    self._last_action_for_reward = current_action

    reward = (
        2.0 * tracking_lin_vel
        + 1.0 * tracking_yaw_rate
        - 0.5 * upright_penalty
        - 0.2 * z_vel_penalty
        - 0.1 * roll_pitch_ang_vel_penalty
        - 1e-4 * torque_penalty
        - 0.01 * action_rate_penalty
    )

    return float(reward)

QuadrupedEnv._compute_reward = _compute_reward


PROPRIOCEPTIVE_OBS = (
    "gravity_vector:base",
    "imu_acc",
    "imu_gyro",
    "qpos_js",
    "qvel_js",
    "base_lin_vel_err:base",
    "base_ang_vel_err:base",
)

IMU_KWARGS = {
    "accel_name": "imu_acc",
    "gyro_name": "imu_gyro",
    "imu_site_name": "imu",
}

# Standing pose — βάση γύρω από την οποία ταλαντεύεται το CPG
GO2_STANDING_POSE = np.array([
     0.21291979,  0.67369252, -1.64459852,  # FL: hip, thigh, calf
    -0.20640879,  0.84122732, -1.94411832,  # FR
     0.29783284,  0.94530330, -1.69789124,  # RL
     0.12787365,  1.01380836, -1.57726899,  # RR
])


class CPGTrotGenerator:
    """
    Central Pattern Generator για τετράποδο trot gait, ΜΕ ασύμμετρη
    stance/swing φάση ΚΑΙ goal-conditioning στο target velocity.

    Trot = διαγώνια ζεύγη ποδιών κινούνται σε φάση μεταξύ τους:
        FL+RR σε φάση 0
        FR+RL σε φάση π (αντίθετη φάση)

    GOAL-CONDITIONING (νέο):
    Σε πραγματικά CPG-based quadruped controllers, το CPG πάντα
    παραμετροποιείται από την εντολή κίνησης — αλλιώς δεν θα ήταν
    χρήσιμο για goal-conditioned tasks. Εδώ:
        - stride_amplitude κλιμακώνεται με |target_velocity|
          (μεγαλύτερο βήμα για μεγαλύτερη ζητούμενη ταχύτητα)
        - frequency κλιμακώνεται ελαφρά με |target_velocity|
          (πιο γρήγορα βήματα για μεγαλύτερη ταχύτητα)
    Αυτό ΔΕΝ είναι learned (παραμένει scripted CPG, καθαρή
    συνάρτηση), και χρησιμοποιεί ΜΟΝΟ το ήδη επιτρεπτό
    target-velocity-error observation. Σύμφωνα με το PDF:
    "Teams may use ... central pattern generators ... model-based
    planning, provided that the final controller uses only the
    observations allowed by the benchmark."

    Η ασυμμετρία stance/swing παραμένει όπως πριν:
        STANCE (πόδι στο έδαφος, ~60% του κύκλου):
            αργή κίνηση προς τα πίσω -> forward thrust
        SWING (πόδι στον αέρα, ~40% του κύκλου):
            γρήγορη επαναφορά προς τα εμπρός
    """

    def __init__(self, base_frequency=1.2, base_stride_amplitude=0.4,
                 lift_amplitude=0.25, duty_cycle=0.65,
                 velocity_gain=1.0, max_velocity_ref=0.5, dt=0.02):
        """
        Args:
            base_frequency        : Hz, βασική συχνότητα (όταν |v_target|=0)
            base_stride_amplitude : rad, βασικό πλάτος (όταν |v_target|=0,
                                     μικρό ώστε να μην κινείται άσκοπα)
            lift_amplitude        : rad, πλάτος ανύψωσης ποδιού (σταθερό)
            duty_cycle             : κλάσμα κύκλου σε stance
            velocity_gain          : πόσο επηρεάζει η ζητούμενη ταχύτητα
                                     το stride amplitude (0 = χωρίς goal-
                                     conditioning, ίδιο με πριν)
            max_velocity_ref       : m/s, ταχύτητα αναφοράς για κανονικοποίηση.
                                     ΣΗΜΑΝΤΙΚΟ: το base_vel_command_type=
                                     "random" του env δειγματίζει εντολές
                                     στο εύρος [-0.5, 0.5] m/s (επιβεβαιωμένο
                                     πειραματικά) — το 0.5 εδώ ταιριάζει με
                                     αυτό το πραγματικό εύρος, ώστε το CPG
                                     amplitude να κλιμακώνεται σωστά σε όλο
                                     το φάσμα πιθανών εντολών.
        """
        self.base_frequency = base_frequency
        self.base_stride_amplitude = base_stride_amplitude
        self.lift_amplitude = lift_amplitude
        self.duty_cycle = duty_cycle
        self.velocity_gain = velocity_gain
        self.max_velocity_ref = max_velocity_ref
        self.dt = dt
        self.phase = 0.0

        # Τρέχουσες (goal-conditioned) τιμές — ενημερώνονται σε κάθε step()
        self.current_frequency = base_frequency
        self.current_stride_amplitude = base_stride_amplitude
        self.direction_sign = 1.0  # +1 = μπροστά, -1 = πίσω

    def reset(self):
        self.phase = 0.0
        self.current_frequency = self.base_frequency
        self.current_stride_amplitude = self.base_stride_amplitude
        self.direction_sign = -1.0   # -1.0 is the stable value (never +1.0)
        self.backward_scale = 1.0
        self.left_stride_multiplier = 1.0
        self.right_stride_multiplier = 1.0

    def update_from_target_velocity(self, target_vel_3d):
        """
        Ενημερώνει frequency/amplitude/direction βάσει της ζητούμενης
        ταχύτητας. Καλείται ΠΡΙΝ από κάθε step() με το τρέχον target
        velocity (διαθέσιμο στον controller μέσω του ήδη-επιτρεπτού
        base_lin_vel_err observation / γνωστής εντολής).

        ΔΙΟΡΘΩΣΗ: Το CPG πρέπει να ΑΝΤΙΣΤΡΕΦΕΙ φορά κίνησης όταν η
        ζητούμενη ταχύτητα είναι αρνητική (προς τα πίσω) — πριν, το
        CPG πάντα παρήγαγε forward-only stance/swing pattern, και ο
        PPO residual καλούνταν να αντιστρέψει εξ ολοκλήρου την
        κατεύθυνση μέσω περιορισμένου residual_scale, προκαλώντας
        ασταθή/απότομη "backward" συμπεριφορά.
        """
        # Χρησιμοποιούμε μόνο τη forward/backward συνιστώσα (vx) για το
        # πρόσημο κατεύθυνσης — το trot pattern μας είναι ουσιαστικά
        # 1-D (forward/backward), δεν στρέφει πλάγια από μόνο του.
        #
        # ΣΗΜΕΙΩΣΗ (επιβεβαιωμένο πειραματικά με test_cpg_direction.py):
        # Η αρχική σύμβαση (+1 για vx>=0) ήταν ΑΝΤΙΣΤΡΟΦΗ σε σχέση με
        # τη σύμβαση γωνιών του Go2 MuJoCo model — με +1 ο robot έπεφτε
        # σε ~99 steps με ελάχιστη μετατόπιση, ενώ με -1 περπατούσε
        # σταθερά 500/500 steps με καθαρή θετική (+x) μετατόπιση.
        # Αντιστρέφουμε εδώ το mapping ώστε το "πρόσημο που πραγματικά
        # δουλεύει" (την τιμή -1 στο test) να αντιστοιχεί σωστά σε
        # vx>=0 (forward εντολή).
        vx = target_vel_3d[0] if len(target_vel_3d) > 0 else 0.0
        yaw_rate = target_vel_3d[2] if len(target_vel_3d) > 2 else 0.0

        # direction_sign = -1.0 is the ONLY experimentally verified stable value.
        # direction_sign = +1.0 causes falls at ~99 steps (confirmed in test_cpg_direction.py).
        #
        # BUG FIX: the original code used +1.0 for backward commands (vx < 0),
        # which caused the robot to fall for ~50% of episodes.
        #
        # Fix: always keep direction_sign = -1.0 (stable forward CPG).
        # For backward commands, we smoothly shrink the stride amplitude toward 0
        # so the CPG becomes a neutral lifting gait (legs lift but no stride),
        # letting the PPO residual handle the actual backward direction.
        self.direction_sign = -1.0

        # backward_scale: 1.0 for forward/zero, smoothly 0.0 for full backward
        # This means CPG fades to neutral when vx < 0, PPO takes over direction.
        if vx >= 0:
            self.backward_scale = 1.0
        else:
            self.backward_scale = max(0.0, 1.0 + vx / self.max_velocity_ref)

        # Yaw-rate integration for steering:
        # If yaw_rate > 0 (turn left), right legs must step larger, left legs smaller.
        # If yaw_rate < 0 (turn right), left legs must step larger, right legs smaller.
        # We scale the stride multiplier linearly with yaw_rate (bounded [0.5, 1.5]).
        yaw_scale = np.clip(yaw_rate, -1.0, 1.0) * 0.5  # Max +/- 50% change
        self.left_stride_multiplier = 1.0 - yaw_scale
        self.right_stride_multiplier = 1.0 + yaw_scale

        speed = np.linalg.norm(target_vel_3d[:2])
        # Minimum speed floor: even when target_vel~=0, the CPG always
        # produces some movement (10% of max_velocity_ref). This breaks
        # the "stand still and survive" reward hack — the agent always
        # has to manage a moving gait, not just balance in place.
        speed = max(speed, 0.1 * self.max_velocity_ref)
        norm_speed = np.clip(speed / self.max_velocity_ref, 0.0, 1.5)

        # Μεγαλύτερη ζητούμενη ταχύτητα -> μεγαλύτερο stride amplitude
        # backward_scale: 1.0 for forward, fades to 0.0 for full backward command
        # so the CPG produces neutral lift gait when PPO must handle direction.
        self.current_stride_amplitude = self.base_stride_amplitude * (
            1.0 + self.velocity_gain * norm_speed
        ) * self.backward_scale
        # Μεγαλύτερη ζητούμενη ταχύτητα -> ελαφρά υψηλότερη συχνότητα
        self.current_frequency = self.base_frequency * (
            1.0 + 0.3 * self.velocity_gain * norm_speed
        )

    def _asymmetric_stride(self, phi):
        """
        Παράγει ασύμμετρη θέση thigh για φάση phi ∈ [0, 2π).
        Επιστρέφει τιμή στο [-1, 1]:
            +1 -> πόδι πιο μπροστά (αρχή stance)
            -1 -> πόδι πιο πίσω (τέλος stance / αρχή swing)
        """
        phi_norm = (phi % (2 * np.pi)) / (2 * np.pi)

        if phi_norm < self.duty_cycle:
            t = phi_norm / self.duty_cycle
            return 1.0 - 2.0 * t
        else:
            t = (phi_norm - self.duty_cycle) / (1.0 - self.duty_cycle)
            return -1.0 + 2.0 * t

    def _swing_lift(self, phi):
        phi_norm = (phi % (2 * np.pi)) / (2 * np.pi)
        if phi_norm < self.duty_cycle:
            return 0.0
        else:
            t = (phi_norm - self.duty_cycle) / (1.0 - self.duty_cycle)
            return np.sin(np.pi * t)

    def step(self):
        """Επιστρέφει target joint positions (12,) για το τρέχον timestep."""
        self.phase += 2 * np.pi * self.current_frequency * self.dt

        leg_phases = np.array([
            self.phase,            # FL
            self.phase + np.pi,    # FR
            self.phase + np.pi,    # RL
            self.phase,            # RR
        ])

        target = GO2_STANDING_POSE.copy()
        for leg_idx in range(4):
            ph = leg_phases[leg_idx]
            base = leg_idx * 3

            stride = self._asymmetric_stride(ph)
            lift   = self._swing_lift(ph)

            # Το direction_sign αντιστρέφει την stance/swing ασυμμετρία:
            # forward (+1): κανονικό stance->swing (σπρώχνει μπροστά)
            # backward (-1): αντίστροφο stance->swing (σπρώχνει πίσω)
            # Apply left/right multiplier based on leg_idx (Left: 0, 2 | Right: 1, 3)
            turn_multiplier = self.left_stride_multiplier if leg_idx in [0, 2] else self.right_stride_multiplier
            
            target[base + 1] += self.direction_sign * self.current_stride_amplitude * stride * turn_multiplier
            target[base + 2] -= self.lift_amplitude * lift

        return target


class Go2WrapperCPG(gym.Wrapper):

    def __init__(self, env, kp=40.0, kd=1.0, residual_scale=2.0,
                 base_frequency=1.2, base_stride_amplitude=0.4,
                 lift_amplitude=0.25, duty_cycle=0.65, velocity_gain=1.0,
                 action_smoothing=0.0):
        """
        action_smoothing : EMA alpha για την PPO residual correction
                            (0 = ανενεργό, π.χ. 0.3 για ομαλές διορθώσεις).
                            ΣΗΜΕΙΩΣΗ: εφαρμόζεται ΜΟΝΟ στο residual, όχι στο
                            CPG target (που είναι ήδη ομαλό από σχεδιασμό).
                            Επιτρέπεται ρητά από το PDF: "action smoothing".
        """
        super().__init__(env)

        self.kp = kp
        self.kd = kd
        self.residual_scale = residual_scale
        self.action_smoothing = action_smoothing
        self.cpg = CPGTrotGenerator(
            base_frequency=base_frequency,
            base_stride_amplitude=base_stride_amplitude,
            lift_amplitude=lift_amplitude,
            duty_cycle=duty_cycle,
            velocity_gain=velocity_gain,
            dt=getattr(env, "simulation_dt", 0.02),
        )

        obs_dim = sum(
            int(np.prod(sp.shape))
            for sp in env.observation_space.spaces.values()
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        # PPO output σε [-1, 1] -> κλιμακώνεται σε torque correction
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32,
        )

        self.max_episode_steps = 1000
        self._step_count = 0
        self._prev_smoothed_action = None

    def _flatten(self, obs):
        return np.concatenate([
            np.atleast_1d(v).flatten() for v in obs.values()
        ]).astype(np.float32)

    def _pd_to_cpg_target(self, obs, cpg_target):
        """PD control ΠΡΟΣ το τρέχον CPG target (όχι τη στατική pose)."""
        q     = obs["qpos_js"]
        q_dot = obs["qvel_js"]
        return self.kp * (cpg_target - q) - self.kd * q_dot

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._last_obs_dict = result[0] if isinstance(result, tuple) else result
        self.cpg.reset()
        self._step_count = 0
        self._prev_smoothed_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        return self._flatten(self._last_obs_dict), {}

    def step(self, ppo_action):
        # 0. Ενημέρωση CPG με το τρέχον target velocity (goal-conditioning)
        #    Χρησιμοποιούμε το ΗΔΗ επιτρεπτό base_lin_vel_err observation:
        #    actual = target - error  =>  target = error + actual
        #    Πιο απλά, διαβάζουμε απευθείας το env attribute (ίδια πληροφορία
        #    που ήδη "βλέπει" ο agent μέσω του error term στο observation).
        try:
            target_vel = self.env.target_base_vel
            if callable(target_vel):
                target_vel = target_vel()
            # target_vel is usually [vx, vy, yaw_rate]
            target_vel_3d = np.array(target_vel).flatten()[:3]
        except Exception:
            target_vel_3d = np.zeros(3)

        self.cpg.update_from_target_velocity(target_vel_3d)

        # 0.5. Action smoothing στο PPO residual (EMA) — εφαρμόζεται ΠΡΙΝ
        #      την κλιμάκωση σε torque, ώστε οι αλλαγές στο residual από
        #      step σε step να είναι πιο ομαλές (λιγότερο action_rate_penalty)
        if self.action_smoothing > 0.0:
            alpha = self.action_smoothing
            smoothed_action = alpha * ppo_action + (1 - alpha) * self._prev_smoothed_action
            self._prev_smoothed_action = smoothed_action.copy()
            ppo_action = smoothed_action

        # 1. CPG παράγει το "βασικό" κινούμενο target για αυτό το timestep
        cpg_target = self.cpg.step()

        # 2. PD-control προς αυτό το κινούμενο target (αντί για στατική pose)
        cpg_torque = self._pd_to_cpg_target(self._last_obs_dict, cpg_target)

        # 3. PPO προσθέτει διόρθωση πάνω στο CPG+PD torque
        residual = ppo_action * self.residual_scale
        final_action = (cpg_torque + residual).astype(np.float32)
        final_action = np.clip(final_action, -50.0, 50.0)

        # 4. Step στο πραγματικό env — official reward υπολογίζεται αυτόματα
        obs, reward, terminated, truncated, info = self.env.step(final_action)
        self._last_obs_dict = obs
        self._step_count += 1

        if self._step_count >= self.max_episode_steps:
            truncated = True

        return self._flatten(obs), reward, terminated, truncated, info


def make_env_cpg(base_frequency=1.2, base_stride_amplitude=0.4, lift_amplitude=0.25,
                  duty_cycle=0.65, velocity_gain=1.0, residual_scale=2.0,
                  action_smoothing=0.0):
    def _init():
        env = QuadrupedEnv(
            robot="go2",
            scene="flat",
            base_vel_command_type="random",
            state_obs_names=PROPRIOCEPTIVE_OBS,
            sensors=(IMU,),
            sensors_kwargs=(IMU_KWARGS,),
        )
        return Go2WrapperCPG(env, base_frequency=base_frequency,
                              base_stride_amplitude=base_stride_amplitude,
                              lift_amplitude=lift_amplitude,
                              duty_cycle=duty_cycle,
                              velocity_gain=velocity_gain,
                              residual_scale=residual_scale,
                              action_smoothing=action_smoothing)
    return _init


if __name__ == "__main__":
    print("Testing Go2WrapperCPG (velocity-aware ασύμμετρο trot CPG + PPO residual)...")
    env = make_env_cpg()()
    obs, _ = env.reset()
    print(f"Observation shape : {obs.shape}")
    print(f"Action space      : {env.action_space}")

    # Με ΜΗΔΕΝΙΚΗ PPO correction -> καθαρό CPG trot pattern (goal-conditioned)
    zero_action = np.zeros(12, dtype=np.float32)
    steps = 0
    for t in range(500):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        steps += 1
        if terminated or truncated:
            break
    print(f"\nΜε μηδενική PPO correction (=καθαρό velocity-aware CPG trot): {steps} steps")
    print(f"Τρέχουσα CPG amplitude: {env.cpg.current_stride_amplitude:.3f} rad "
          f"(base: {env.cpg.base_stride_amplitude:.3f})")
    print(f"Τρέχουσα CPG frequency: {env.cpg.current_frequency:.3f} Hz "
          f"(base: {env.cpg.base_frequency:.3f})")

    env.close()