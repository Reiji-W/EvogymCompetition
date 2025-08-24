# server/custom_env/register.py
import os, re, logging
from typing import Optional
import gymnasium as gym
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

# ---- 定数（唯一の source of truth） ------------------------------------------
DEFAULT_MAX_EPISODE_STEPS = 1000

# ---- saved_data から MyJsonWorld-N の最大を見つけて次を採番 -------------------
def _next_env_id_from_saved(saved_root: str = "server/saved_data") -> str:
    try:
        entries = os.listdir(saved_root)
    except FileNotFoundError:
        return "MyJsonWorld-1"

    pat = re.compile(r"^MyJsonWorld-(\d+)$")
    mx = 0
    for name in entries:
        m = pat.match(name)
        if m:
            try:
                mx = max(mx, int(m.group(1)))
            except ValueError:
                pass
    return f"MyJsonWorld-{mx + 1 if mx >= 1 else 1}"

# ---- 既登録チェック -----------------------------------------------------------
def _already_registered(env_id: str) -> bool:
    try:
        gym.spec(env_id)
        return True
    except Exception:
        return False

# ---- 外部公開API --------------------------------------------------------------
def set_default_max_episode_steps(steps: int) -> None:
    """グローバル既定値を変更（MuLambdaES などから呼ぶ）。"""
    global DEFAULT_MAX_EPISODE_STEPS
    DEFAULT_MAX_EPISODE_STEPS = int(steps)

def ensure_registered(
    env_id: Optional[str] = None,
    *,
    max_episode_steps: Optional[int] = None,
    saved_root: str = "server/saved_data",
) -> str:
    """
    - env_id を省略すると saved_data の MyJsonWorld-N を見て次番号で自動採番
    - 既に登録済みなら何もしない（冪等）
    - max_episode_steps を指定しなければ DEFAULT_MAX_EPISODE_STEPS を使用
    - entry_point は env_core._ActiveJsonWalkerEnv
    """
    eid = env_id or _next_env_id_from_saved(saved_root)
    if _already_registered(eid):
        return eid

    steps = int(max_episode_steps) if max_episode_steps is not None else int(DEFAULT_MAX_EPISODE_STEPS)
    register(
        id=eid,
        entry_point="server.custom_env.env_core:_ActiveJsonWalkerEnv",
        max_episode_steps=steps,
    )
    logger.debug(f"[EvoGym register] Registered gym env: {eid} (max_episode_steps={steps})")
    return eid