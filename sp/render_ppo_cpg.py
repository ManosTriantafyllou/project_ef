"""
render_ppo.py
-------------
Φορτώνει το trained PPO model και το τρέχει στο MuJoCo viewer.

Χρήση:
    python render_ppo.py
    python render_ppo.py --model models/best/best_model.zip
    python render_ppo.py --model models_fs4/best/best_model.zip --frame_stack 4
    python render_ppo.py --model models_as0.3/best/best_model.zip --vecnormalize models_as0.3/vecnormalize.pkl --action_smoothing 0.3
"""

import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from env_wrapper import make_env
    from train_ppo import ActionSmoothingWrapper
except ImportError:
    make_env = None
    ActionSmoothingWrapper = None


def make_env_with_stacking(frame_stack=1, action_smoothing=0.0, use_cpg=False,
                            base_frequency=1.2, base_stride_amplitude=0.4, lift_amplitude=0.25,
                            duty_cycle=0.65, velocity_gain=1.0, residual_scale=2.0,
                            cpg_action_smoothing=0.0):
    """ΠΡΕΠΕΙ να ταιριάζει ΑΚΡΙΒΩΣ με το setup του training, αλλιώς το
    trained model βλέπει διαφορετική δυναμική/observations από αυτή με
    την οποία εκπαιδεύτηκε."""
    if use_cpg:
        from env_wrapper_cpg import make_env_cpg
        return make_env_cpg(base_frequency, base_stride_amplitude, lift_amplitude,
                             duty_cycle, velocity_gain, residual_scale,
                             cpg_action_smoothing)

    def _init():
        env = make_env()()
        if action_smoothing > 0.0:
            env = ActionSmoothingWrapper(env, alpha=action_smoothing)
        if frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
            env = gym.wrappers.FlattenObservation(env)
        return env
    return _init


def run(model_path, vecnormalize_path, n_episodes=5, frame_stack=1, action_smoothing=0.0,
        use_cpg=False, base_frequency=1.2, base_stride_amplitude=0.4, lift_amplitude=0.25,
        duty_cycle=0.65, velocity_gain=1.0, residual_scale=2.0, cpg_action_smoothing=0.0):
    print(f"Φόρτωση model    : {model_path}")
    print(f"Frame stack      : {frame_stack}")
    print(f"Action smoothing : {action_smoothing}")
    print(f"CPG mode         : {use_cpg}")
    if use_cpg:
        print(f"CPG action smooth: {cpg_action_smoothing}")

    vec_env = DummyVecEnv([make_env_with_stacking(
        frame_stack, action_smoothing, use_cpg,
        base_frequency, base_stride_amplitude, lift_amplitude, duty_cycle,
        velocity_gain, residual_scale, cpg_action_smoothing
    )])

    try:
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print("VecNormalize φορτώθηκε")
    except FileNotFoundError:
        print(f"{vecnormalize_path} δεν βρέθηκε, τρέχει χωρίς normalization")

    model = PPO.load(model_path, env=vec_env)
    print("Model φορτώθηκε\n")

    # Πρόσβαση στο underlying QuadrupedEnv για render(), ανεξάρτητα από
    # πόσα wrapper layers (FrameStack/Flatten/Monitor/Go2Wrapper) υπάρχουν
    # ανάμεσα — ψάχνουμε για το πρώτο obj που έχει .render().
    base_env = vec_env.envs[0]
    while not hasattr(base_env, "render") or not callable(getattr(base_env, "render", None)):
        if hasattr(base_env, "env"):
            base_env = base_env.env
        else:
            break
    # Στην πράξη το Go2Wrapper/QuadrupedEnv έχουν .render() στην αλυσίδα .env
    render_target = vec_env.envs[0]
    for _ in range(6):  # ξεφλουδίζουμε ως 6 επίπεδα wrappers max
        if hasattr(render_target, "render"):
            try:
                render_target.render()
                break
            except Exception:
                pass
        render_target = getattr(render_target, "env", render_target)

    for ep in range(n_episodes):
        obs = vec_env.reset()

        total_reward = 0.0
        steps = 0

        print(f"Episode {ep+1}/{n_episodes}")

        for t in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            steps += 1

            render_target.render()

            if done[0]:
                break

        print(f"  Steps: {steps} | Reward: {total_reward:.2f}\n")

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best/best_model.zip")
    parser.add_argument("--vecnormalize", default=None,
                         help="Path στο vecnormalize.pkl. Default: <models_dir>/vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--frame_stack", type=int, default=1,
                         help="ΠΡΕΠΕΙ να ταιριάζει με το frame_stack που χρησιμοποιήθηκε στο training!")
    parser.add_argument("--action_smoothing", type=float, default=0.0,
                         help="ΠΡΕΠΕΙ να ταιριάζει με το action_smoothing που χρησιμοποιήθηκε στο training!")
    parser.add_argument("--cpg", action="store_true",
                         help="Χρήση CPG wrapper (για models_cpg/* checkpoints)")
    parser.add_argument("--base_frequency", type=float, default=1.2)
    parser.add_argument("--base_stride_amplitude", type=float, default=0.4)
    parser.add_argument("--lift_amplitude", type=float, default=0.25)
    parser.add_argument("--duty_cycle", type=float, default=0.65)
    parser.add_argument("--velocity_gain", type=float, default=1.0)
    parser.add_argument("--residual_scale", type=float, default=2.0)
    parser.add_argument("--cpg_action_smoothing", type=float, default=0.0,
                         help="ΠΡΕΠΕΙ να ταιριάζει με το --action_smoothing του train_ppo_cpg.py!")
    args = parser.parse_args()

    if args.vecnormalize is None:
        import os
        models_dir = os.path.dirname(os.path.dirname(args.model.rstrip("/\\")))
        if not models_dir:
            models_dir = "models"
        args.vecnormalize = os.path.join(models_dir, "vecnormalize.pkl")

    run(args.model, args.vecnormalize, args.episodes, args.frame_stack, args.action_smoothing,
        args.cpg, args.base_frequency, args.base_stride_amplitude, args.lift_amplitude,
        args.duty_cycle, args.velocity_gain, args.residual_scale, args.cpg_action_smoothing)