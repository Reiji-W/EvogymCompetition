# server/trainer/ga/engine.py
from __future__ import annotations

import os
import glob
import shutil
from typing import Optional, Tuple

import gymnasium as gym
import evogym.envs  # noqa: F401  # ensure default EvoGym envs are registered

from server.custom_env import ensure_registered

# gym.spec(entry_point) の判定でカスタム環境かどうかを見分けるための目印
_CUSTOM_ENTRY_SUBSTRINGS = ("custom_env.env_core", "server.custom_env.env_core")
_ACTIVE_JSON_ENVVAR = "EVOGYM_ACTIVE_JSON_OVERRIDE"


def resolve_env(
    env_name: Optional[str],
    max_episode_steps: Optional[int],
    *,
    force_custom: bool = False,
) -> Tuple[str, bool]:
    """
    使う環境IDを決定して返す。
    返り値: (env_id, is_custom)
      - force_custom=True: env_name を無視して active JSON を使い、custom_env の登録フローを使用
      - env_name が未指定: ensure_registered(None, ...) で次番号を採番・登録し、(eid, True)
      - 既存の spec がある:
          entry_point に custom_env の文字列が含まれていればカスタムとみなし ensure_registered で（必要なら）再登録
          それ以外はベース環境として (env_name, False)
      - spec が見つからない: カスタムとして ensure_registered し、(eid, True)
    """
    if force_custom:
        # custom モードでは env_name を無視して active JSON を必ず使う。
        # これにより、--env_name が Walker-v0 等でもベース環境に落ちない。
        active_json = _find_active_json_path()
        eid = ensure_registered(None, world_json=active_json, max_episode_steps=max_episode_steps)
        return eid, True

    if not env_name:
        # 未指定 → カスタムとして自動採番・登録
        return ensure_registered(None, max_episode_steps=max_episode_steps), True

    try:
        spec = gym.spec(env_name)
        entry = str(getattr(spec, "entry_point", ""))
        is_custom = any(s in entry for s in _CUSTOM_ENTRY_SUBSTRINGS)
        if is_custom:
            # 既にカスタムとして登録済み or 同名カスタムを使いたいケース
            eid = ensure_registered(env_name, max_episode_steps=max_episode_steps)
            return eid, True
        else:
            # ベース環境：上書き不要
            return env_name, False
    except Exception:
        # 未登録 → カスタムとして登録
        eid = ensure_registered(env_name, max_episode_steps=max_episode_steps)
        return eid, True


def _find_active_json_path() -> str:
    """
    server/custom_env/active 配下から JSON を1件だけ見つけて絶対パスを返す。
    複数・ゼロ件はエラー。
    """
    override = os.environ.get(_ACTIVE_JSON_ENVVAR)
    if override:
        override = os.path.abspath(override)
        if os.path.isfile(override):
            return override

    base = os.path.abspath(os.path.join("server", "custom_env", "active"))
    jsons = sorted(glob.glob(os.path.join(base, "*.json")))
    if len(jsons) != 1:
        raise RuntimeError(
            f"active JSON は 1 件のみ必要です（検出 {len(jsons)} 件；{base}）。"
        )
    return os.path.abspath(jsons[0])


def copy_active_assets(home_path: str, env_id: Optional[str] = None) -> None:
    """
    実験ディレクトリ（home_path）に「実際に使う world JSON」を同梱する。
    再現性確保のため：
      - env_id が与えられていて spec.kwargs["world_json"] があればそれを優先
      - それ以外は active/（もしくは EVOGYM_ACTIVE_JSON_OVERRIDE）から 1 件を解決
      - JSON を <home_path>/ にコピー
      - JSON の親ディレクトリ名（active or worlds）配下に、その JSON だけをコピー
    """
    world_json: Optional[str] = None
    if env_id:
        try:
            spec = gym.spec(env_id)
            kwargs = getattr(spec, "kwargs", None) or {}
            candidate = kwargs.get("world_json")
            if candidate and os.path.isfile(candidate):
                world_json = os.path.abspath(candidate)
        except Exception:
            world_json = None

    if world_json is None:
        world_json = _find_active_json_path()

    # JSON をルート直下にコピー
    shutil.copy2(world_json, os.path.join(home_path, os.path.basename(world_json)))

    # 親ディレクトリ名（active or worlds）を作って、その JSON だけコピー
    src_dir = os.path.dirname(world_json)
    dst_name = os.path.basename(src_dir)
    dst_dir = os.path.join(home_path, dst_name)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(world_json, os.path.join(dst_dir, os.path.basename(world_json)))


__all__ = ["resolve_env", "copy_active_assets"]
