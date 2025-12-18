# server/custom_env/register.py
import os, re, logging, shutil
from typing import Optional
import gymnasium as gym
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

# ---- 定数（唯一の source of truth） ------------------------------------------
DEFAULT_MAX_EPISODE_STEPS = 1000
_ACTIVE_JSON_ENVVAR = "EVOGYM_ACTIVE_JSON_OVERRIDE"

def _custom_env_dir(*parts: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *parts)

def _worlds_dir() -> str:
    return _custom_env_dir("worlds")

def _world_json_path_for_env_id(env_id: str) -> str:
    return os.path.join(_worlds_dir(), f"{env_id}.json")

# ---- MyJsonWorld-N の最大を見つけて次を採番 ----------------------------------
def _next_env_id(saved_root: str = "server/saved_data") -> str:
    entries: list[str] = []

    # 旧仕様: saved_data のディレクトリ名からも拾う
    try:
        entries.extend(os.listdir(saved_root))
    except FileNotFoundError:
        pass

    # 新仕様: server/custom_env/worlds/MyJsonWorld-N.json を source of truth にする
    try:
        for name in os.listdir(_worlds_dir()):
            entries.append(name)
    except FileNotFoundError:
        pass

    pat = re.compile(r"^MyJsonWorld-(\d+)$")
    pat_json = re.compile(r"^MyJsonWorld-(\d+)\.json$")
    mx = 0
    for name in entries:
        m = pat.match(name) or pat_json.match(name)
        if m:
            try:
                mx = max(mx, int(m.group(1)))
            except ValueError:
                pass
    return f"MyJsonWorld-{mx + 1 if mx >= 1 else 1}"

# ---- active/override から JSON を1件に解決 ------------------------------------
def _resolve_active_json_for_registration() -> str:
    override = os.environ.get(_ACTIVE_JSON_ENVVAR)
    if override:
        override = os.path.abspath(override)
        if not os.path.isfile(override):
            raise RuntimeError(f"EVOGYM_ACTIVE_JSON_OVERRIDE が指す JSON が見つかりません: {override}")
        return override

    active_dir = _custom_env_dir("active")
    if not os.path.isdir(active_dir):
        raise RuntimeError(f"active ディレクトリが見つかりません: {active_dir}")

    jsons = sorted(p for p in (os.path.join(active_dir, f) for f in os.listdir(active_dir)) if p.endswith(".json"))
    if len(jsons) != 1:
        raise RuntimeError(f"active 配下の JSON は 1 件のみ必要です（検出 {len(jsons)} 件；{active_dir}）。")
    return os.path.abspath(jsons[0])

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
    world_json: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    saved_root: str = "server/saved_data",
) -> str:
    """
    - env_id を省略すると saved_data の MyJsonWorld-N を見て次番号で自動採番
    - 既に登録済みなら何もしない（冪等）
    - max_episode_steps を指定しなければ DEFAULT_MAX_EPISODE_STEPS を使用
    - entry_point は env_core._ActiveJsonWalkerEnv
    - world_json を指定すると、その JSON を env_id に紐付けて登録（kwargs で固定）
    """
    eid = env_id or _next_env_id(saved_root)
    if _already_registered(eid):
        return eid

    os.makedirs(_worlds_dir(), exist_ok=True)
    pinned_world_json = _world_json_path_for_env_id(eid)

    # 既に worlds/<env_id>.json があるなら、それを「正」として使う（spawn 等でも再現可能）
    if os.path.isfile(pinned_world_json):
        if world_json is not None:
            req = os.path.abspath(world_json)
            if os.path.realpath(req) != os.path.realpath(pinned_world_json):
                raise RuntimeError(
                    f"{eid} は既に {pinned_world_json} に紐付いています。別の JSON を使うには別の env_id を使ってください。"
                )
    else:
        src = os.path.abspath(world_json) if world_json else _resolve_active_json_for_registration()
        if not os.path.isfile(src):
            raise RuntimeError(f"world_json が見つかりません: {src}")
        shutil.copy2(src, pinned_world_json)

    steps = int(max_episode_steps) if max_episode_steps is not None else int(DEFAULT_MAX_EPISODE_STEPS)
    register(
        id=eid,
        entry_point="server.custom_env.env_core:_ActiveJsonWalkerEnv",
        kwargs={"world_json": pinned_world_json},
        max_episode_steps=steps,
    )
    logger.debug(f"[EvoGym register] Registered gym env: {eid} (max_episode_steps={steps})")
    return eid
