# server/trainer/ga/engine.py
from __future__ import annotations

import os
import glob
import shutil
from typing import Optional, Tuple

import gymnasium as gym

from server.custom_env import ensure_registered

# gym.spec(entry_point) の判定でカスタム環境かどうかを見分けるための目印
_CUSTOM_ENTRY_SUBSTRINGS = ("custom_env.env_core", "server.custom_env.env_core")


def resolve_env(env_name: Optional[str], max_episode_steps: Optional[int]) -> Tuple[str, bool]:
    """
    使う環境IDを決定して返す。
    返り値: (env_id, is_custom)
      - env_name が未指定: ensure_registered(None, ...) で次番号を採番・登録し、(eid, True)
      - 既存の spec がある:
          entry_point に custom_env の文字列が含まれていればカスタムとみなし ensure_registered で（必要なら）再登録
          それ以外はベース環境として (env_name, False)
      - spec が見つからない: カスタムとして ensure_registered し、(eid, True)
    """
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
    # env_core が既にロードされている場合は、その中の _ACTIVE_JSON を優先
    try:
        import server.custom_env.env_core as core  # type: ignore
        p = getattr(core, "_ACTIVE_JSON", None)
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    except Exception:
        # 読み込み失敗時はディレクトリ走査にフォールバック
        pass

    base = os.path.abspath(os.path.join("server", "custom_env", "active"))
    jsons = sorted(glob.glob(os.path.join(base, "*.json")))
    if len(jsons) != 1:
        raise RuntimeError(
            f"active JSON は 1 件のみ必要です（検出 {len(jsons)} 件；{base}）。"
        )
    return os.path.abspath(jsons[0])


def copy_active_assets(home_path: str) -> None:
    """
    実験ディレクトリ（home_path）に active JSON と active/ ディレクトリを同梱する。
    再現性確保のため：
      - active/*.json のうち使用中の1件を <home_path>/ にコピー
      - server/custom_env/active ディレクトリ全体を <home_path>/active にコピー
        （既にある場合は削除してからコピー）
    """
    active_json = _find_active_json_path()
    # JSON をルート直下にコピー
    shutil.copy2(active_json, os.path.join(home_path, os.path.basename(active_json)))

    # active ディレクトリを丸ごとコピー
    active_dir_src = os.path.dirname(active_json)  # server/custom_env/active
    active_dir_dst = os.path.join(home_path, "active")
    if os.path.isdir(active_dir_dst):
        shutil.rmtree(active_dir_dst)
    shutil.copytree(active_dir_src, active_dir_dst)


__all__ = ["resolve_env", "copy_active_assets"]