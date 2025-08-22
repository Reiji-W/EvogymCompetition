# server/custom_env/register.py
"""
Active JSON-based EvoGym environment registrar.

- server/custom_env/active 配下に .json が「ちょうど1つ」あることを要求。
- 0件または2件以上なら、警告を表示して登録を中止（RuntimeError を投げる）。
- 1件のみの場合、その JSON を読み込む環境を定義・登録。
  登録ID: "MyJsonWorld-Walker-v0"

使い方:
  import server.custom_env.register
  # import した時点で登録が走るので、
  # gym.make("MyJsonWorld-Walker-v0", body=..., connections=..., render_mode=...) が使用可能。
"""

import os
import sys
import glob
import numpy as np
from typing import Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from evogym import EvoWorld
from evogym.envs import EvoGymBase
from gymnasium import spaces
import logging
logger = logging.getLogger(__name__)

# --- Active JSON 検査 ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ACTIVE_DIR = os.path.join(_THIS_DIR, "active")

if not os.path.isdir(_ACTIVE_DIR):
    msg = (
        f"[EvoGym register] active ディレクトリが見つかりません: {_ACTIVE_DIR}\n"
        f"server/custom_env/active を作成し、使用するワールド JSON を 1 件だけ置いてください。"
    )
    print(msg, file=sys.stderr)
    raise RuntimeError(msg)

_jsons = sorted(glob.glob(os.path.join(_ACTIVE_DIR, "*.json")))
if len(_jsons) != 1:
    msg = (
        "[EvoGym register] active 配下の JSON は 1 件のみである必要があります。\n"
        f"  見つかった件数: {len(_jsons)}\n"
        f"  検出リスト:\n" + "".join(f"    - {p}\n" for p in _jsons) +
        "※ active には使用したい JSON を 1 つだけ残し、他は移動/削除してください。"
    )
    print(msg, file=sys.stderr)
    raise RuntimeError(msg)

_ACTIVE_JSON = os.path.abspath(_jsons[0])
logger.debug(f"[EvoGym register] Using active JSON: {_ACTIVE_JSON}")


# --- 環境定義（Active JSON を使用） ---
class _ActiveJsonWalkerEnv(EvoGymBase):
    """
    active JSON を読み込み、渡された body/connections を配置して動かす環境。
    - 前進量を報酬
    - 落下死亡判定（kill_y より下が一定ステップ続いたら終了）
    """

    def __init__(
        self,
        body: np.ndarray,
        connections: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        *,
        spawn_x: int = 1,
        spawn_y: int = 18,
        action_low: float = 0.6,
        action_high: float = 1.6,
        # ↓ 追加: 落下関連パラメータ
        kill_y: float = 0.5,           # これより下に一定ステップ居続けると死亡
        kill_grace_steps: int = 10,    # 連続で下回ったら終了とみなす猶予
        fall_penalty: float = 5.0,     # 終了時に与えるペナルティ（報酬から減点）
    ):
        self._kill_y = float(kill_y)
        self._kill_grace_steps = int(kill_grace_steps)
        self._fall_penalty = float(fall_penalty)
        self._below_counter = 0

        # ワールド読み込み（active JSON）
        world = EvoWorld.from_json(_ACTIVE_JSON)
        world.add_from_array("robot", body, x=spawn_x, y=spawn_y, connections=connections)

        super().__init__(world, render_mode=render_mode)

        # アクション／観測空間
        n_act = self.get_actuator_indices("robot").size
        obs = self._get_obs()
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=obs.shape, dtype=np.float32)

        if hasattr(self, "default_viewer"):
            self.default_viewer.track_objects("robot")

    def _get_obs(self) -> np.ndarray:
        vel = self.get_vel_com_obs("robot")
        rel = self.get_relative_pos_obs("robot")
        return np.asarray(np.concatenate([vel, rel], axis=0), dtype=np.float32)

    def step(self, action):
        # 前進量を素朴な報酬に
        pos1 = self.object_pos_at_time(self.get_time(), "robot")
        done_parent = super().step({"robot": action})
        pos2 = self.object_pos_at_time(self.get_time(), "robot")

        com1 = np.mean(pos1, axis=1)
        com2 = np.mean(pos2, axis=1)
        reward = float(com2[0] - com1[0])

        # ---- 落下死亡判定 ----
        # pos2 は shape (2, N) 想定 → y成分の最小値でチェック
        y_min = float(np.min(pos2[1]))
        if y_min < self._kill_y:
            self._below_counter += 1
        else:
            self._below_counter = 0

        fell = self._below_counter >= self._kill_grace_steps
        if fell:
            reward -= self._fall_penalty

        obs = self._get_obs()
        info = {"y_min": y_min, "fell": fell}
        terminated = bool(done_parent or fell)
        truncated = False
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._below_counter = 0
        obs = self._get_obs()
        info = {}
        return obs, info
# --- Gym に登録 ---
# 既存と互換の ID を使います（必要に応じて変更可）
ENV_ID = "MyJsonWorld-Walker-v0"

# すでに登録済みならスキップ（再登録エラーを避ける）
if ENV_ID not in gym.envs.registry:
    register(
        id=ENV_ID,
        entry_point=f"{__name__}:_ActiveJsonWalkerEnv",
    )
    logger.debug(f"[EvoGym register] Registered gym env: {ENV_ID}")
else:
    logger.debug(f"[EvoGym register] Env already registered: {ENV_ID}")