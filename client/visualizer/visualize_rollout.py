#!/usr/bin/env python3
# client/visualizer/visualize_rollout.py
import os
import sys
import argparse
import math
import numpy as np
import gymnasium as gym
from typing import Optional

# ---- path bootstrap: project root を import path に通す ----
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# ---- warnings: evogym import 前に抑制 ----
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---- evogym/env の登録 ----
import evogym.envs  # base envs を有効化

def _register_custom_env():
    """custom_env をどこに置いても拾えるように動的 import"""
    for mod in ("custom_env", "server.custom_env", "client.custom_env"):
        try:
            __import__(mod)
            return
        except Exception:
            continue
_register_custom_env()

# ── 周期的アクチュエーション関数 ─────────────────────────────
def periodic_controller(
    step: int,
    num_actuators: int,
    controller_params: Optional[tuple] = None,
    *,
    frequency: float = 0.1,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    if controller_params is not None:
        frequency, amplitude, phase = controller_params
    value = amplitude * math.sin(2 * math.pi * frequency * step + phase)
    return np.full((num_actuators,), value, dtype=np.float32)

# ── 直接ステップ実行ロールアウト ───────────────────────────────
def rollout_direct(
    env_name: str,
    n_iters: int,
    body: np.ndarray,
    connections: Optional[np.ndarray] = None,
    controller_params: Optional[tuple] = None,
):
    env = gym.make(env_name, body=body, connections=connections, render_mode="human")
    # 必要なら最大ステップ上書き
    try:
        env._max_episode_steps = max(getattr(env, "_max_episode_steps", 0) or 0, n_iters)
    except Exception:
        pass

    obs, _ = env.reset()
    total_reward = 0.0

    for step in range(n_iters):
        action = periodic_controller(step, env.action_space.shape[0], controller_params=controller_params)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"\nTotal reward: {total_reward:.5f}\n")
    env.close()

# ── コードサイン実験の可視化 ───────────────────────────────────
def visualize_codesign(env_name: str, saved_root: str, exp_name: str):
    EXP_DIR = os.path.join(saved_root, exp_name)
    if not os.path.isdir(EXP_DIR):
        raise FileNotFoundError(f"saved_root/exp が見つかりません: {EXP_DIR}")

    # 世代フォルダの番号を収集
    gen_list = sorted(
        int(d.split("_")[1])
        for d in os.listdir(EXP_DIR)
        if d.startswith("generation_")
    )

    # 全世代の output.txt から (gen, rank, label, reward) を収集
    all_robots = []
    for gen in gen_list:
        out_path = os.path.join(EXP_DIR, f"generation_{gen}", "output.txt")
        if not os.path.exists(out_path):
            continue
        with open(out_path) as f:
            rank = 1
            for line in f:
                parts = line.split()
                label = int(parts[0])
                reward = float(parts[1])
                all_robots.append((gen, rank, label, reward))
                rank += 1

    # 報酬でソートして上位を表示
    all_robots.sort(key=lambda x: x[3], reverse=True)
    top_n = min(30, len(all_robots))
    print("\n=== Top Robots Across Generations ===")
    for i in range(top_n):
        gen, rank, label, rew = all_robots[i]
        print(f"{i+1:2d}: gen {gen} | rank {rank} | ID {label} | reward {rew:.4f}")
    print()

    # 対話的選択ループ
    while True:
        try:
            gen_number = int(input("Enter generation number: "))
        except ValueError:
            print("数字を入力してください。")
            continue
        out_path = os.path.join(EXP_DIR, f"generation_{gen_number}", "output.txt")
        if not os.path.exists(out_path):
            print("無効な世代番号です。再入力してください。")
            continue

        # 選択世代の全ロボットを列挙
        gen_data = []
        with open(out_path) as f:
            rank = 1
            for line in f:
                parts = line.split()
                label = int(parts[0])
                rew = float(parts[1])
                gen_data.append((rank, label, rew))
                rank += 1

        print(f"\n=== Generation {gen_number} Robots ===")
        for rank, label, rew in gen_data:
            print(f"{rank:2d}: ID {label} | reward {rew:.4f}")
        print()

        inp = input("Enter robot rank(s) (e.g. 1 or 1-3 or 1,4,5): ")
        selected = []
        for part in inp.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    a, b = map(int, part.split("-", 1))
                    selected.extend(range(a, b + 1))
                except ValueError:
                    pass
            else:
                try:
                    selected.append(int(part))
                except ValueError:
                    pass

        try:
            num_iters = int(input("Enter num iters: "))
        except ValueError:
            print("ステップ数は整数で入力してください。")
            continue
        print()

        # 各選択ロボットを可視化
        for rank in selected:
            if rank < 1 or rank > len(gen_data):
                print(f"無効なランク: {rank}")
                continue
            _, label, _ = gen_data[rank - 1]

            struct_path = os.path.join(
                EXP_DIR,
                f"generation_{gen_number}",
                "structure",
                f"{label}.npz",
            )
            if not os.path.exists(struct_path):
                print(f"構造データが見つかりません: {struct_path}")
                continue
            data = np.load(struct_path)
            body = data["arr_0"]
            connections = data["arr_1"]
            controller_params = None
            if "arr_2" in data.files:
                arr2 = data["arr_2"]
                controller_params = tuple(arr2.tolist()) if hasattr(arr2, "tolist") else tuple(arr2)

            print(f"\n--- Visualizing Gen{gen_number} Rank{rank} (ID {label}) ---")
            print(f"Body shape: {body.shape}, # connections: {connections.shape[1]}")

            if num_iters > 0:
                rollout_direct(env_name, num_iters, body, connections, controller_params=controller_params)
        print()

def main():
    parser = argparse.ArgumentParser(description="EvoGym コードサイン結果の可視化 (周期的アクチュエータ制御のみ)")
    parser.add_argument("--env-name", default="MyJsonWorld-Walker-v0", help="EvoGym 環境名 (例: Walker-v0)")
    parser.add_argument(
        "--saved-root",
        default=os.path.join("client/mnt"),
        help="結果ディレクトリのルート（例: server/saved_data または リモートマウント先）",
    )
    args = parser.parse_args()

    # 実験一覧
    if not os.path.isdir(args.saved_root):
        print(f"saved_root が見つかりません: {args.saved_root}")
        return

    saved = sorted(d for d in os.listdir(args.saved_root) if os.path.isdir(os.path.join(args.saved_root, d)))
    if not saved:
        print("実験が見つかりません。まずサーバで実行してください。")
        return

    print("Available experiments:")
    for e in saved:
        print(f"  {e}")
    exp_name = input("\nEnter experiment name: ").strip()
    while exp_name not in saved:
        exp_name = input("無効な名前です。再入力してください: ").strip()

    visualize_codesign(args.env_name, args.saved_root, exp_name)

if __name__ == "__main__":
    main()