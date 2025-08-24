#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
μ+λ Evolution Strategy for EvoGym
- 1 世代あたり「生存 μ 個 + 子 λ 個 = 集団サイズ pop_size」
- 並列で評価し、saved_data/<exp>/generation_<g>/ に保存
- 直接実行:  python server/trainer/MuLambdaES.py
"""

from __future__ import annotations

import os
import sys
import math
import shutil
import random
import argparse
from typing import List, Tuple, Optional

import numpy as np
import gymnasium as gym
from pathlib import Path
import importlib
import glob

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- パスと環境登録 ----------------------------------------------------
PROJECT_ROOT = os.path.abspath("./")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import evogym.envs
import server.custom_env  # noqa: F401
from server.custom_env import ensure_registered

from evogym import sample_robot, hashable
from server.trainer.utils.algo_utils import (
    Structure,
    mutate,
    get_percent_survival_evals,
)
from server.trainer.utils.mp_group import Group

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

def _mp_bootstrap_register():
    try:
        proj_root = str(Path(__file__).resolve().parents[2])
    except Exception:
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    importlib.invalidate_caches()
    import server.custom_env.register  # noqa: F401

# ────────────────────────────────────────────────────────────────
# コントローラ: 単一周波数の正弦波（f, a, p）
def periodic_controller(step: int, n_act: int, params: Tuple[float, float, float]) -> np.ndarray:
    f, a, p = params
    val = a * math.sin(2 * math.pi * f * step + p)
    return np.full((n_act,), val, dtype=np.float32)

def evaluate_structure(
    body: np.ndarray,
    connections: np.ndarray,
    controller_params: Tuple[float, float, float],
    env_name: str,
    max_steps: int,
) -> float:
    _mp_bootstrap_register()
    env = gym.make(env_name, body=body, connections=connections, render_mode=None)
    obs, _ = env.reset()
    total = 0.0
    for t in range(max_steps):
        action = periodic_controller(t, env.action_space.shape[0], controller_params)
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
            f_str = f"{s.controller_params[0]:.4f},{s.controller_params[1]:.4f},{s.controller_params[2]:.4f}"
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

    env_id = ensure_registered(env_name, max_episode_steps=max_episode_steps)

    home_path = os.path.join("server/saved_data", exp_name)
    if os.path.exists(home_path):
        shutil.rmtree(home_path)
    os.makedirs(home_path, exist_ok=True)

    active_json = _find_active_json_path()
    shutil.copy2(active_json, os.path.join(home_path, os.path.basename(active_json)))

    with open(os.path.join(home_path, "metadata.txt"), "w") as f:
        f.write("ALGO: mu+lambda ES\n")
        f.write("ENV: {}\n".format(env_id))
        f.write(f"POP_SIZE: {pop_size}\n")
        f.write(f"STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n")
        f.write(f"MAX_EVALUATIONS: {max_evaluations}\n")
        f.write(f"MAX_STEPS: {max_steps}\n")

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
    parser.add_argument("--exp_name", type=str, default="es_exp3", help="実験名（saved_data/ 配下）")
    parser.add_argument("--env_name", type=str, default=None, help="環境 ID（None なら自動採番）")
    parser.add_argument("--pop_size", type=int, default=120, help="集団サイズ (μ+λ)")
    parser.add_argument("--structure_shape", type=int, nargs=2, default=[5, 5], help="構造サイズ (W H)")
    parser.add_argument("--max_evaluations", type=int, default=12000, help="最大評価回数（新規個体数の上限）")
    parser.add_argument("--num_cores", type=int, default=12, help="並列評価プロセス数")
    parser.add_argument("--max_steps", type=int, default=100000, help="1 個体あたりの最大ステップ数")
    parser.add_argument("--max_episode_steps", type=int, default=None, help="環境のエピソード最大ステップ数（未指定時はデフォルトを使用）")
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