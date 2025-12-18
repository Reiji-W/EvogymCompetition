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
_ACTIVE_JSON_ENVVAR = "EVOGYM_ACTIVE_JSON_OVERRIDE"

def _resolve_active_json() -> str:
    """
    active ディレクトリの JSON もしくは EVOGYM_ACTIVE_JSON_OVERRIDE で指定された JSON を返す。
    override が与えられている場合は active ディレクトリの存在チェックをスキップする。
    """
    override = os.environ.get(_ACTIVE_JSON_ENVVAR)
    if override:
        override = os.path.abspath(override)
        if not os.path.isfile(override):
            msg = (
                "[EvoGym] EVOGYM_ACTIVE_JSON_OVERRIDE で指定された JSON が見つかりません。\n"
                f"  指定パス: {override}"
            )
            print(msg, file=sys.stderr)
            raise RuntimeError(msg)
        return override

    if not os.path.isdir(_ACTIVE_DIR):
        msg = (
            f"[EvoGym] active ディレクトリが見つかりません: {_ACTIVE_DIR}\n"
            "server/custom_env/active に使用するワールド JSON を1件だけ置いてください。"
        )
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

    jsons = sorted(glob.glob(os.path.join(_ACTIVE_DIR, "*.json")))
    if len(jsons) != 1:
        msg = (
            "[EvoGym] active 配下の JSON は 1 件のみである必要があります。\n"
            f"  見つかった件数: {len(jsons)}\n"
            "  検出リスト:\n" + "".join(f"    - {p}\n" for p in jsons)
        )
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

    return os.path.abspath(jsons[0])


_ACTIVE_JSON: Optional[str] = None

def _get_default_world_json() -> str:
    global _ACTIVE_JSON
    if _ACTIVE_JSON and os.path.isfile(_ACTIVE_JSON):
        return _ACTIVE_JSON
    _ACTIVE_JSON = _resolve_active_json()
    logger.debug(f"[EvoGym] Using active JSON: {_ACTIVE_JSON}")
    return _ACTIVE_JSON

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
        world_json: Optional[str] = None,
        *,
        # 初期位置
        spawn_x: int = 1,
        spawn_y: int = 18,
        # アクチュエータのスケール
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

        world_path = os.path.abspath(world_json) if world_json else _get_default_world_json()
        if not os.path.isfile(world_path):
            msg = (
                "[EvoGym] world_json が見つかりません。\n"
                f"  world_json: {world_path}\n"
                "  env_id ごとの JSON を使う場合は register() の kwargs で world_json を渡してください。"
            )
            print(msg, file=sys.stderr)
            raise RuntimeError(msg)

        world = EvoWorld.from_json(world_path)
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
        reward = float(com2[0] - com1[0]) # x 方向の前進量

        # 落下判定: ロボの下端が連続kill_grace_steps回 kill_y 未満なら終了 & ペナルティ
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
