# server/custom_env/env_core.py
import os, sys, glob, logging
from typing import Optional
import numpy as np

from evogym import EvoWorld
from evogym.envs import EvoGymBase
from gymnasium import spaces

logger = logging.getLogger(__name__)

# -- active JSON を1件だけ要求 -------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ACTIVE_DIR = os.path.join(_THIS_DIR, "active")

if not os.path.isdir(_ACTIVE_DIR):
    msg = (
        f"[EvoGym] active ディレクトリが見つかりません: {_ACTIVE_DIR}\n"
        "server/custom_env/active に使用するワールド JSON を1件だけ置いてください。"
    )
    print(msg, file=sys.stderr)
    raise RuntimeError(msg)

_jsons = sorted(glob.glob(os.path.join(_ACTIVE_DIR, "*.json")))
if len(_jsons) != 1:
    msg = (
        "[EvoGym] active 配下の JSON は 1 件のみである必要があります。\n"
        f"  見つかった件数: {len(_jsons)}\n"
        "  検出リスト:\n" + "".join(f"    - {p}\n" for p in _jsons)
    )
    print(msg, file=sys.stderr)
    raise RuntimeError(msg)

_ACTIVE_JSON = os.path.abspath(_jsons[0])
logger.debug(f"[EvoGym] Using active JSON: {_ACTIVE_JSON}")

# -- 環境本体（落下死亡判定などのロジック） --------------------------------------
class _ActiveJsonWalkerEnv(EvoGymBase):
    """
    active の JSON を読み込み、渡された body/connections を配置して動かす最小環境。
    - 報酬: 前進量 (COM の x 変化)
    - 落下死亡判定: y_min が kill_y を連続 kill_grace_steps 下回ると終了 & ペナルティ
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
        # 落下系
        kill_y: float = 0.5,
        kill_grace_steps: int = 10,
        fall_penalty: float = 5.0,
    ):
        self._kill_y = float(kill_y)
        self._kill_grace_steps = int(kill_grace_steps)
        self._fall_penalty = float(fall_penalty)
        self._below_counter = 0

        world = EvoWorld.from_json(_ACTIVE_JSON)
        world.add_from_array("robot", body, x=spawn_x, y=spawn_y, connections=connections)
        super().__init__(world, render_mode=render_mode)

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
        pos1 = self.object_pos_at_time(self.get_time(), "robot")
        done_parent = super().step({"robot": action})
        pos2 = self.object_pos_at_time(self.get_time(), "robot")

        com1 = np.mean(pos1, axis=1)
        com2 = np.mean(pos2, axis=1)
        reward = float(com2[0] - com1[0])

        y_min = float(np.min(pos2[1]))
        self._below_counter = (self._below_counter + 1) if y_min < self._kill_y else 0
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