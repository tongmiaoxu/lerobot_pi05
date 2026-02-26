import cma
import os
import mujoco
import time
from pathlib import Path
from mujoco import MjModel, MjData
import numpy as np
from datetime import datetime
import pickle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
from lerobot_mujoco_utils import GRIPPER_OPEN_MM, lerobot_state_to_mujoco_ctrl

FPS = 30.0
TIMESTEP = 1.0 / FPS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
XML_DIR = _PROJECT_ROOT / "xarm7"
PARQUET_PATH = _PROJECT_ROOT / "data" / "data" / "chunk-000" / "file-000.parquet"

# xarm7 XML defaults per actuator class:
#   size1 (j1,j2):    gainprm=1500  biasprm=[0,-1500,-150]  dof_damping=10
#   size2 (j3,j4,j5): gainprm=1000  biasprm=[0,-1000,-100]  dof_damping=5
#   size3 (j6,j7):    gainprm=800   biasprm=[0, -800, -80]  dof_damping=2
#
# Parameter vector layout (21-dim):
#   [ kp × 7 | act_damping × 7 | jnt_damping × 7 ]
#
# Applied as:
#   gainprm[0]  =  kp            (position gain / stiffness)
#   biasprm[1]  = -kp            (keeps PD structure)
#   biasprm[2]  = -act_damping   (actuator velocity damping)
#   dof_damping =  jnt_damping   (passive joint damping)

KP_INIT = np.array([1500.0, 1500, 1000, 1000, 1000, 800, 800])
KP_LOW = np.ones(7)
KP_HIGH = np.array([3000.0, 3000, 2000, 2000, 2000, 1600, 1600])

ACT_DAMP_INIT = np.array([150.0, 150, 100, 100, 100, 80, 80])
ACT_DAMP_LOW = np.zeros(7)
ACT_DAMP_HIGH = np.array([500.0, 500, 300, 300, 300, 250, 250])

JNT_DAMP_INIT = np.array([10.0, 10, 5, 5, 5, 2, 2])
JNT_DAMP_LOW = np.zeros(7)
JNT_DAMP_HIGH = np.array([50.0, 50, 30, 30, 30, 15, 15])


def save_trace_numpy(trace, filename):
    """Save CMA trace to a .npz file."""
    if not filename.endswith('.npz'):
        filename += '.npz'

    trace_arrays = {
        'm': np.array([t['m'] for t in trace]),
        'sigma': np.array([t['σ'] for t in trace]),
        'C': np.array([t['C'] for t in trace]),
        'p_sigma': np.array([t['p_σ'] for t in trace]),
        'p_C': np.array([t['p_C'] for t in trace]),
        'B': np.array([t['B'] for t in trace]),
        'D': np.array([t['D'] for t in trace]),
        'population': np.array([t['population'] for t in trace])
    }

    np.savez(filename, **trace_arrays)


def load_trace_numpy(filename):
    """Load CMA trace from a .npz file."""
    if not filename.endswith('.npz'):
        filename += '.npz'

    data = np.load(filename)

    trace = []
    for i in range(len(data['m'])):
        trace.append({
            'm': data['m'][i],
            'σ': data['sigma'][i],
            'C': data['C'][i],
            'p_σ': data['p_sigma'][i],
            'p_C': data['p_C'][i],
            'B': data['B'][i],
            'D': data['D'][i],
            'population': data['population'][i]
        })

    return trace


def load_episodes(parquet_path):
    """Load all episodes from a LeRobot v3.0 parquet file.

    Returns dict mapping episode_idx -> observation.state array (N, 8)
    where values are [j1..j7 in degrees, gripper in mm].
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    episodes = {}
    for ep_idx in sorted(df["episode_index"].unique()):
        ep = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        obs = np.stack(ep["observation.state"].values).astype(np.float64)
        episodes[ep_idx] = obs
    return episodes


def main():
    start = time.time()

    # Load xarm7 MuJoCo model (chdir needed so mesh paths resolve)
    original_cwd = os.getcwd()
    try:
        os.chdir(str(XML_DIR))
        model = MjModel.from_xml_path("scene.xml")
    finally:
        os.chdir(original_cwd)
    data = MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    gripper_act_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"
    )
    gripper_mj_range = (
        model.actuator_ctrlrange[gripper_act_id, 0],
        model.actuator_ctrlrange[gripper_act_id, 1],
    )

    print(f"[INFO] Loading dataset from {PARQUET_PATH}")
    raw_episodes = load_episodes(str(PARQUET_PATH))
    print(f"[INFO] Loaded {len(raw_episodes)} episode(s), "
          f"total frames: {sum(len(v) for v in raw_episodes.values())}")

    # Pre-convert all episodes: degrees/mm -> radians/mujoco-gripper
    episodes_mj = {
        idx: lerobot_state_to_mujoco_ctrl(obs, gripper_mj_range)
        for idx, obs in raw_episodes.items()
    }

    def evaluate_trajectory(prms):
        """Evaluate how well simulation tracks real xArm observations."""
        kp = prms[:7]
        act_damp = prms[7:14]
        jnt_damp = prms[14:]

        score = 0
        L = 0

        for obs_mj in episodes_mj.values():
            N = len(obs_mj)
            L += N

            mujoco.mj_resetDataKeyframe(model, data, home_id)

            model.actuator_gainprm[:7, 0] = kp
            model.actuator_biasprm[:7, 1] = -kp
            model.actuator_biasprm[:7, 2] = -act_damp
            model.dof_damping[:7] = jnt_damp

            data.qpos[:7] = obs_mj[0, :7]
            mujoco.mj_step(model, data)

            for i in range(N):
                err = np.linalg.norm(obs_mj[i, :7] - data.qpos[:7])
                score += err

                data.ctrl[:] = obs_mj[i]
                curr_sim_time = data.time
                while data.time < curr_sim_time + TIMESTEP:
                    mujoco.mj_step(model, data)

        return score / L if L > 0 else float('inf')

    # ------------------------------------------------------------------
    max_epochs = 1000

    curr_state = np.concatenate([KP_INIT, ACT_DAMP_INIT, JNT_DAMP_INIT])
    lower_bounds = np.concatenate([KP_LOW, ACT_DAMP_LOW, JNT_DAMP_LOW])
    upper_bounds = np.concatenate([KP_HIGH, ACT_DAMP_HIGH, JNT_DAMP_HIGH])

    print(f"Parameter vector: {len(curr_state)} dims  "
          f"(7 kp + 7 act_damp + 7 jnt_damp)")
    print("STARTING SEARCH")

    es = cma.CMAEvolutionStrategy(
        curr_state,
        20.0,
        {
            'bounds': [lower_bounds.tolist(), upper_bounds.tolist()],
            'maxiter': max_epochs,
            'verbose': 1,
        }
    )

    generation = 0
    prev_best = float('inf')
    stall_counter = 0
    patience = 20
    tol = 1e-6
    while not es.stop():
        population = es.ask()
        fitness_values = [evaluate_trajectory(c) for c in population]
        es.tell(population, fitness_values)
        curr_best = es.result.fbest
        if abs(prev_best - curr_best) < tol:
            stall_counter += 1
        else:
            stall_counter = 0
        if stall_counter >= patience:
                print("Converged after plateau.")
                break

        prev_best = curr_best
        if generation % 10 == 0:
            print(f"Generation {generation}: best fitness = {es.result.fbest:.6f}")
            print(f"Best solution: {es.result.xbest}")
            print(f"Best fitness: {es.result.fbest}")
        generation += 1

    best_solution = es.result.xbest
    best_fitness = es.result.fbest
    elapsed = time.time() - start
    print(f"search done  ({elapsed:.1f}s)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("./logs_cmaes", exist_ok=True)

    log_path = os.path.join("./logs_cmaes", f"cma_results_{timestamp}.txt")
    with open(log_path, "w") as f:
        f.write("CMA-ES Optimization Results (xArm7)\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n\n")
        f.write(f"Best Fitness Score: {best_fitness}\n\n")
        f.write("Best Solution:\n")
        f.write(f"Stiffness kp   (gainprm):   {best_solution[:7].tolist()}\n")
        f.write(f"Act damping b2 (biasprm[2]): {best_solution[7:14].tolist()}\n")
        f.write(f"Jnt damping d  (dof_damping): {best_solution[14:].tolist()}\n")

    with open("cma_result.pkl", "wb") as f:
        pickle.dump({
            'xbest': es.result.xbest,
            'fbest': es.result.fbest,
            'evals_best': es.result.evals_best,
            'evaluations': es.result.evaluations,
            'iterations': es.result.iterations,
            'stds': es.result.stds,
        }, f)

    print(f"Results saved to: {log_path}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting CMA-ES optimization for xArm7...")
    main()
    logger.info("CMA-ES optimization completed.")
