# server/custom_env/envs/__init__.py
from gymnasium.envs.registration import register
import gymnasium as gym

ENV_ID = "MyJsonWorld-Walker-v0"

def _already_registered(env_id: str) -> bool:
    try:
        gym.spec(env_id)  # 登録済みなら取れる
        return True
    except Exception:
        return False

if not _already_registered(ENV_ID):
    register(
        id=ENV_ID,
        entry_point="server.custom_env.register:_ActiveJsonWalkerEnv",  # ← ここを現在のパスに！
        max_episode_steps=100000,
    )