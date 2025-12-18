from __future__ import annotations
import math
from pathlib import Path
import importlib
import sys

import gymnasium as gym
import numpy as np

# --- マルチプロセス時の環境登録（元の挙動と同等） -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAPPED = False

def _mp_bootstrap_register() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    importlib.invalidate_caches()
    import evogym.envs  # noqa: F401  # ensure default EvoGym envs are registered
    import server.custom_env.register  # noqa: F401
    _BOOTSTRAPPED = True

# --- コントローラ（元の f,a,p 正弦波） ---------------------------------------
def periodic_controller(step: int, n_act: int, params: tuple[float, float, float]) -> np.ndarray:
    f, a, p = params
    val = a * math.sin(2 * math.pi * f * step + p)
    return np.full((n_act,), val, dtype=np.float32)

# --- フォールバック判定（本当に引数不正のときだけ） ---------------------------
def _should_fallback(e: Exception) -> bool:
    msg = str(e).lower()
    return ("unexpected keyword" in msg) or ("got an unexpected keyword" in msg)

# --- 評価本体：累積報酬を返す（元のロジックを忠実に） -------------------------
def evaluate_structure(
    body: np.ndarray,
    connections: np.ndarray,
    controller_params: tuple[float, float, float],
    env_name: str,
    max_steps: int,
) -> float:
    _mp_bootstrap_register()

    # spawn の場合に備えて、必要なら env_id を（worlds/<env_id>.json に基づいて）登録する。
    try:
        gym.spec(env_name)
    except Exception:
        from server.custom_env import ensure_registered
        ensure_registered(env_name)

    # まず body/conn 付きで作成（元の意図）。引数不正のときだけフォールバック。
    try:
        env = gym.make(env_name, body=body, connections=connections, render_mode=None)
    except (TypeError, gym.error.Error) as e:
        if _should_fallback(e):
            env = gym.make(env_name, render_mode=None)
        else:
            raise

    # アクチュエータ数チェック（0 なら異常として明示）
    shape = getattr(env.action_space, "shape", None)
    if not shape or shape[0] <= 0:
        env.close()
        raise RuntimeError(f"action_space が不正（shape={shape}）。body/conn を受け取る環境になっていない可能性。")

    # 元コード準拠：controller_params は丸めずそのまま使う
    params = controller_params

    obs, _ = env.reset()
    total = 0.0
    n_act = shape[0]
    for t in range(max_steps):
        action = periodic_controller(t, n_act, params)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break

    env.close()
    return total
