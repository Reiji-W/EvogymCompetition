#!/usr/bin/env python3
# client/visualizer/visualize_rollout.py
# 可視化は client/mnt 直下を対話選択（従来どおり）
# 環境登録だけ学習時と同じ server/custom_env/env_core.py / register.py を使う

import os
import sys
import argparse
import math
import traceback
import numpy as np
import gymnasium as gym
from typing import Optional
import faulthandler

faulthandler.enable()

# === project root を sys.path に通す（学習側と同じ） ===
import importlib
PROJECT_ROOT = os.path.abspath("./")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
importlib.invalidate_caches()

ACTIVE_JSON_ENVVAR = "EVOGYM_ACTIVE_JSON_OVERRIDE"

# === custom_env の登録機構を使う（学習時と同じモジュールを利用） ===
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import evogym.envs  # base envs を有効化（学習時と同様）

def _register_custom_env():
    """custom_env をどこに置いても拾えるように動的 import（後方互換）"""
    for mod in ("custom_env", "server.custom_env", "client.custom_env"):
        try:
            importlib.import_module(mod)
            return mod
        except Exception:
            continue
    return None

# ── metadata と JSON で環境を用意する ─────────────────────────
def _read_metadata_env(exp_dir: str) -> Optional[str]:
    meta = os.path.join(exp_dir, "metadata.txt")
    if not os.path.isfile(meta):
        return None
    env_id = None
    with open(meta, "r") as f:
        for line in f:
            if line.startswith("ENV:"):
                env_id = line.split(":", 1)[1].strip()
                break
    return env_id or None

def _find_bundled_json(exp_dir: str) -> Optional[str]:
    # run_es で active JSON を EXP_DIR 直下に copy2 済み（カスタム環境の場合）
    jsons = [p for p in os.listdir(exp_dir) if p.lower().endswith(".json")]
    if len(jsons) == 0:
        # ベース環境や、JSON を同梱していない古い実験に対応するため None を返す
        return None
    if len(jsons) != 1:
        raise RuntimeError(
            f"実験ディレクトリ直下の JSON は 0 または 1 件のみを想定しています（検出 {len(jsons)} 件；{exp_dir}）。"
        )
    return os.path.abspath(os.path.join(exp_dir, jsons[0]))

def _ensure_env_from_bundle(exp_dir: str, cli_env_fallback: Optional[str]) -> str:
    """
    可視化でも学習時と同じ環境を再現する:
      - metadata.txt の ENV を優先（なければ CLI の --env-name）
      - （カスタム時のみ）実験直下に同梱された JSON を env_core の _ACTIVE_JSON に差し替え（存在する場合）
      - ensure_registered(ENV) で登録
    """
    env_id = _read_metadata_env(exp_dir) or (cli_env_fallback or "MyJsonWorld-Local")

    # env_id が既に登録済みか / カスタムかを判定
    try:
        spec = gym.spec(env_id)
        entry = str(getattr(spec, "entry_point", ""))
        is_custom = ("custom_env.env_core" in entry) or ("server.custom_env.env_core" in entry)
        already_registered = True
    except Exception:
        # 見つからない＝学習で使ったカスタム想定（登録が必要）
        entry = ""
        is_custom = True
        already_registered = False

    if not is_custom:
        # Walker-v0 など既存ベース環境なら、そのまま使う
        print(f"[DEBUG] '{env_id}' is a base env (entry_point={entry}). Skip bundled JSON and snapshot.")
        return env_id

    # --- ここから先は custom_env のときだけ実行 ---
    bundled_json = _find_bundled_json(exp_dir)
    if bundled_json is not None:
        os.environ[ACTIVE_JSON_ENVVAR] = bundled_json

    snap_root = os.path.join(exp_dir, "code_snapshot")
    if os.path.isdir(snap_root) and snap_root not in sys.path:
        sys.path.insert(0, snap_root)
        importlib.invalidate_caches()

    # snapshot 側の custom_env を import
    mod_name = _register_custom_env()
    if mod_name is None:
        raise ImportError("custom_env パッケージが import できません。")

    core = importlib.import_module(f"{mod_name}.env_core")
    reg  = importlib.import_module(f"{mod_name}.register")

    # 実験同梱 JSON を使用（存在する場合だけ）
    if bundled_json is not None:
        setattr(core, "_ACTIVE_JSON", bundled_json)

    ensure_registered = getattr(reg, "ensure_registered")
    # max_episode_steps は学習時に依存。指定しない（None）。
    ensure_registered(env_id, max_episode_steps=None)

    if bundled_json is not None:
        print(f"[DEBUG] Bundled JSON set to: {bundled_json}")
    print(f"[DEBUG] Registered/validated ENV: {env_id}")
    return env_id

# ── 周期的アクチュエータ（学習時と同じ） ─────────────────────────
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

# ── 直接ステップ実行（学習時と一致：オフセット/クリップ無し） ───────────
def rollout_direct(
    env_name: str,
    n_iters: int,
    body: np.ndarray,
    connections: Optional[np.ndarray] = None,
    controller_params: Optional[tuple] = None,
):
    # まずはカスタム環境想定（body/conn 必須）で作成。ダメならベース環境として作成。
    env = None
    print("[DEBUG] gym.make(render_mode='human') start", flush=True)
    try:
        env = gym.make(env_name, body=body, connections=connections, render_mode="human")
        print("[DEBUG] gym.make(render_mode='human') succeeded", flush=True)
    except TypeError as e:
        print(f"[DEBUG] gym.make(render_mode='human') raised TypeError -> retry render_mode=None ({e})", flush=True)
        try:
            env = gym.make(env_name, render_mode=None)
            print("[DEBUG] gym.make(render_mode=None) succeeded", flush=True)
        except Exception:
            print("[ERROR] gym.make(render_mode=None) failed", flush=True)
            traceback.print_exc()
            return
    except Exception:
        print("[ERROR] gym.make(render_mode='human') failed", flush=True)
        traceback.print_exc()
        return

    try:
        env._max_episode_steps = max(getattr(env, "_max_episode_steps", 0) or 0, n_iters)
    except Exception:
        pass

    # デバッグ表示（必要なら）
    try:
        print("[DEBUG] action_space:", getattr(env, "action_space", None))
        print("[DEBUG] n_actuators:", getattr(env, "action_space", None).shape[0])
    except Exception:
        pass
    if controller_params is not None:
        f, a, p = controller_params
        print(f"[DEBUG] controller_params f={f:.6f}, a={a:.6f}, p={p:.6f}")
    else:
        print("[DEBUG] controller_params: None (default periodic_controller)")
    print("[DEBUG] env.reset() start", flush=True)
    try:
        obs, _ = env.reset()
        print("[DEBUG] env.reset() succeeded", flush=True)
    except Exception:
        print("[ERROR] env.reset() failed", flush=True)
        traceback.print_exc()
        env.close()
        return
    total_reward = 0.0
    for step in range(n_iters):
        action = periodic_controller(step, env.action_space.shape[0], controller_params)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            print(f"[DEBUG] early_stop at step={step} (terminated={terminated}, truncated={truncated})")
            break

    print(f"\nTotal reward: {total_reward:.5f}\n")
    env.close()

# ── 可視化（client/mnt 直下を対話選択・従来どおり） ──────────────────────
def visualize_codesign(env_name_cli: str, saved_root: str, exp_name: str):
    EXP_DIR = os.path.join(saved_root, exp_name)
    if not os.path.isdir(EXP_DIR):
        raise FileNotFoundError(f"saved_root/exp が見つかりません: {EXP_DIR}")

    # ★ metadata の ENV と実験直下 JSON を使って、学習時と同じ環境を登録
    env_id = _ensure_env_from_bundle(EXP_DIR, env_name_cli)

    # 世代一覧
    gen_list = sorted(
        int(d.split("_")[1])
        for d in os.listdir(EXP_DIR)
        if d.startswith("generation_")
    )

    # 各世代の output.txt を読み込み、(gen, rank, label, reward)
    all_robots = []
    for gen in gen_list:
        out_path = os.path.join(EXP_DIR, f"generation_{gen}", "output.txt")
        if not os.path.exists(out_path):
            continue
        with open(out_path) as f:
            rank = 1
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                label = int(parts[0])
                reward = float(parts[1])
                all_robots.append((gen, rank, label, reward))
                rank += 1

    all_robots.sort(key=lambda x: x[3], reverse=True)
    top_n = min(30, len(all_robots))
    print("\n=== Top Robots Across Generations ===")
    for i in range(top_n):
        gen, rank, label, rew = all_robots[i]
        print(f"{i+1:2d}: gen {gen} | rank {rank} | ID {label} | reward {rew:.4f}")
    print()

    # 対話ループ（従来通り）
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

        gen_data = []
        with open(out_path) as f:
            rank = 1
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
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

        for rank in selected:
            if rank < 1 or rank > len(gen_data):
                print(f"無効なランク: {rank}")
                continue
            _, label, _ = gen_data[rank - 1]
            struct_path = os.path.join(
                EXP_DIR, f"generation_{gen_number}", "structure", f"{label}.npz"
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
            print(f"Using ENV: {env_id}")

            if num_iters > 0:
                rollout_direct(env_id, num_iters, body, connections, controller_params=controller_params)
        print()

def main():
    parser = argparse.ArgumentParser(description="EvoGym 可視化（環境は学習時と同一の登録を使用）")
    parser.add_argument("--env-name", default=None,
                        help="環境ID（metadata があればそちらを優先）")
    parser.add_argument("--saved-root", default=os.path.join("client", "mnt"),
                        help="可視化対象のルート（client/mnt 直下を列挙・従来どおり）")
    args = parser.parse_args()

    # 実験一覧は client/mnt 直下（従来どおり）
    if not os.path.isdir(args.saved_root):
        print(f"saved_root が見つかりません: {args.saved_root}")
        return

    saved = sorted(d for d in os.listdir(args.saved_root)
                   if os.path.isdir(os.path.join(args.saved_root, d)))
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
