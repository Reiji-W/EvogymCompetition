# server/trainer/MuLambdaES.py
from __future__ import annotations

# ── Standard library
import os
import sys
import math
import shutil
import random
import argparse
import importlib
import glob
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

# モジュール実行を直接実行で代替するため（server/ を見せる）
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # MuLambdaES.py は server/trainer/ にある前提
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
importlib.invalidate_caches()

# ── Third-party
import numpy as np
import gymnasium as gym

# ── Local imports（ここより後で OK）
from server.custom_env import ensure_registered
from server.trainer.utils.mp_group import Group
from evogym import sample_robot, hashable
from server.trainer.utils.algo_utils import (
    Structure,
    mutate,
    get_percent_survival_evals,
)

warnings.filterwarnings("ignore", category=UserWarning)
# 先頭の import 群の下あたりに追加
_CUSTOM_ENTRY_SUBSTRINGS = ("custom_env.env_core", "server.custom_env.env_core")

def _resolve_env(env_name: Optional[str], max_episode_steps: Optional[int]) -> Tuple[str, bool]:
    """env_id を決定し、(env_id, is_custom) を返す。
    - 既存のベース環境なら ensure_registered は呼ばない（上書き防止）
    - 見つからない/カスタムなら ensure_registered で登録
    """
    if not env_name:
        # 未指定ならカスタムを採番・登録
        return ensure_registered(None, max_episode_steps=max_episode_steps), True

    # 既存 spec を調べる
    try:
        spec = gym.spec(env_name)
        entry = str(getattr(spec, "entry_point", ""))
        is_custom = any(s in entry for s in _CUSTOM_ENTRY_SUBSTRINGS)
        if is_custom:
            # すでにカスタムとして登録されている or 同名カスタムを使いたい
            eid = ensure_registered(env_name, max_episode_steps=max_episode_steps)
            return eid, True
        else:
            # ベース環境：上書きせず、そのまま使う
            return env_name, False
    except Exception:
        # 未登録（＝学習で使うのはカスタム想定）
        eid = ensure_registered(env_name, max_episode_steps=max_episode_steps)
        return eid, True
    
# _ACTIVE_JSON を探す。JSONが複数ある場合は停止。
def _find_active_json_path() -> str:
    try:
        import server.custom_env.env_core as core
        p = getattr(core, "_ACTIVE_JSON", None)
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    except Exception:
        pass

    base = os.path.abspath(os.path.join("server", "custom_env", "active"))
    jsons = sorted(glob.glob(os.path.join(base, "*.json")))
    if len(jsons) != 1:
        raise RuntimeError(
            f"active JSON は 1 件のみ必要です（検出 {len(jsons)} 件；{base}）。"
        )
    return os.path.abspath(jsons[0])

# マルチプロセスで環境登録を行うためのブートストラップ
_BOOTSTRAPPED = False  # 各プロセス内での冪等化用
def _mp_bootstrap_register() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    importlib.invalidate_caches()
    import server.custom_env.register
    _BOOTSTRAPPED = True

# ────────────────────────────────────────────────────────────────
# コントローラ🎮️: 単一周波数の正弦波（f, a, p）
def periodic_controller(
    step: int,         # 現在のステップ数（時間を整数で表す）
    n_act: int,        # アクチュエータの数（行動ベクトルの次元）
    params: Tuple[float, float, float]  # (f, a, p) 周波数, 振幅, 位相
) -> np.ndarray:
    f, a, p = params
    val = a * math.sin(2 * math.pi * f * step + p)
    return np.full((n_act,), val, dtype=np.float32)

# 評価関数📈
def evaluate_structure(
    body: np.ndarray,
    connections: np.ndarray,
    controller_params: Tuple[float, float, float],
    env_name: str,
    max_steps: int,
) -> float:
    _mp_bootstrap_register()
    # 既に run_es 側で解決済みの env_name が来る前提。
    # まずは body/conn 付きで試す（カスタム想定）。TypeError 等ならベース環境として再試行。
    try:
        env = gym.make(env_name, body=body, connections=connections, render_mode=None)
    except TypeError:
        env = gym.make(env_name, render_mode=None)
    except gym.error.Error:
        # 一部の実装は gym.error に包むので同様にフォールバック
        env = gym.make(env_name, render_mode=None)  
    # 丸め誤差対策で float32 にする
    params32 = tuple(np.asarray(controller_params, dtype=np.float32).tolist())
    obs, _ = env.reset()
    total = 0.0
    for t in range(max_steps):
        action = periodic_controller(
            t, env.action_space.shape[0], params32
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    env.close()
    return total

def save_generation(
    home_path: str,
    generation: int,
    structures: List["ESIndividual"],
) -> None:
    gen_dir = os.path.join(home_path, f"generation_{generation}")
    struct_dir = os.path.join(gen_dir, "structure")
    os.makedirs(struct_dir, exist_ok=True)
    with open(os.path.join(gen_dir, "output.txt"), "w") as fout:
        for s in structures:
            np.savez(
                os.path.join(struct_dir, f"{s.label}.npz"),
                s.body,
                s.connections,
                np.array(s.controller_params, dtype=np.float32),
            )
            f_str = (
                f"{s.controller_params[0]:.4f},"
                f"{s.controller_params[1]:.4f},"
                f"{s.controller_params[2]:.4f}"
            )
            fout.write(f"{s.label}\t{s.fitness:.4f}\t{f_str}\n")


class ESIndividual(Structure):
    def __init__(
        self,
        body: np.ndarray,
        connections: np.ndarray,
        label: int,
        controller_params: Optional[Tuple[float, float, float]] = None,
    ):
        super().__init__(body, connections, label)
        if controller_params is None:
            self.controller_params = (
                random.uniform(0.01, 0.07),
                random.uniform(0.5, 1.5),
                random.uniform(0.0, 2 * math.pi),
            )
        else:
            self.controller_params = controller_params

    def mutate_child(self, new_label: int) -> Optional["ESIndividual"]:
        child = mutate(self.body.copy(), mutation_rate=0.1, num_attempts=50)
        if child is None:
            return None
        body_c, conn_c = child
        f, a, p = self.controller_params
        if random.random() < 0.2:
            f = max(0.001, min(f * random.uniform(0.8, 1.2), 0.2))
        if random.random() < 0.3:
            a = max(0.2, min(a * random.uniform(0.8, 1.2), 2.0))
        if random.random() < 0.3:
            p += random.uniform(-0.5, 0.5)
        return ESIndividual(body_c, conn_c, new_label, (f, a, p))


def run_es(
    exp_name: str,
    env_name: Optional[str],
    pop_size: int,
    structure_shape: Tuple[int, int],
    max_evaluations: int,
    num_cores: int,
    max_steps: int,
    max_episode_steps: Optional[int] = None,
) -> None:
    env_id, _ = _resolve_env(env_name, max_episode_steps)

    home_path = os.path.join("server/saved_data", exp_name)
    if os.path.exists(home_path):
        shutil.rmtree(home_path)
    os.makedirs(home_path, exist_ok=True)

    active_json = _find_active_json_path()
    shutil.copy2(
        active_json, os.path.join(home_path, os.path.basename(active_json))
    )

    # env_core.py, register.py のスナップショットを保存
    snap_dir = os.path.join(home_path, "code_snapshot", "custom_env")
    os.makedirs(snap_dir, exist_ok=True)
    for fname in ("env_core.py", "register.py"):
        src = os.path.join("server", "custom_env", fname)
        if os.path.isfile(src):
            dst = os.path.join(snap_dir, fname)
            shutil.copy2(src, dst)


    with open(os.path.join(home_path, "metadata.txt"), "w") as f:
        f.write("ALGO: mu+lambda ES\n")
        f.write("ENV: {}\n".format(env_id))
        f.write(f"POP_SIZE: {pop_size}\n")
        f.write(f"STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n")
        f.write(f"MAX_EVALUATIONS: {max_evaluations}\n")
        f.write(f"MAX_STEPS: {max_steps}\n")
        try:
            import evogym, gymnasium, numpy as _np

            f.write(
                f"VERSIONS: evogym={getattr(evogym, '__version__', 'unknown')}, "
                f"gymnasium={getattr(gymnasium, '__version__', 'unknown')}, "
                f"numpy={_np.__version__}\n"
            )
        except Exception:
            pass

    structures: List[ESIndividual] = []
    seen_hashes = set()
    num_evals = 0
    gen = 0

    for i in range(pop_size):
        body, connections = sample_robot(structure_shape)
        while hashable(body) in seen_hashes:
            body, connections = sample_robot(structure_shape)
        structures.append(ESIndividual(body, connections, i))
        seen_hashes.add(hashable(body))
        num_evals += 1

    while num_evals <= max_evaluations:
        print(f"Generation {gen} | evals {num_evals}/{max_evaluations}")

        group = Group()
        for s in structures:
            group.add_job(
                evaluate_structure,
                (s.body, s.connections, s.controller_params, env_id, max_steps),
                callback=s.set_reward,
            )
        group.run_jobs(num_cores)

        structures.sort(key=lambda x: x.fitness, reverse=True)
        save_generation(home_path, gen, structures)

        if num_evals >= max_evaluations:
            break

        pct = get_percent_survival_evals(num_evals, max_evaluations)
        mu = max(2, math.ceil(pop_size * pct))
        lam = max(1, pop_size - mu)

        survivors = structures[:mu]
        for idx, s in enumerate(survivors):
            s.is_survivor = True
            s.prev_gen_label = s.label
            s.label = idx

        children: List[ESIndividual] = []
        next_label = mu
        while len(children) < lam and num_evals < max_evaluations:
            parent = random.choice(survivors)
            child = parent.mutate_child(next_label)
            if child is None:
                continue
            h = hashable(child.body)
            if h in seen_hashes:
                continue
            children.append(child)
            seen_hashes.add(h)
            next_label += 1
            num_evals += 1

        structures = survivors + children
        gen += 1

    print("ES complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="μ+λ ES for EvoGym")
    parser.add_argument(
        "--exp_name", type=str, default="noseed", help="実験名（saved_data/ 配下）"
    )
    parser.add_argument(
        "--env_name", type=str, default=None, help="環境 ID（None なら自動採番）"
    )
    parser.add_argument(
        "--pop_size", type=int, default=120, help="集団サイズ (μ+λ)"
    )
    parser.add_argument(
        "--structure_shape",
        type=int,
        nargs=2,
        default=[5, 5],
        help="構造サイズ (W H)",
    )
    parser.add_argument(
        "--max_evaluations",
        type=int,
        default=1000,
        help="最大評価回数（新規個体数の上限）",
    )
    parser.add_argument(
        "--num_cores", type=int, default=1, help="並列評価プロセス数"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="1 個体あたりの最大ステップ数"
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=None,
        help="環境のエピソード最大ステップ数（未指定時はデフォルトを使用）",
    )
    args = parser.parse_args()

    run_es(
        exp_name=args.exp_name,
        env_name=args.env_name,
        pop_size=args.pop_size,
        structure_shape=tuple(args.structure_shape),
        max_evaluations=args.max_evaluations,
        num_cores=args.num_cores,
        max_steps=args.max_steps,
        max_episode_steps=args.max_episode_steps,
    )