#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
軽量並列実行ユーティリティ
- API: Group.add_job(func, args, callback), Group.run_jobs(num_cores)
- 例外は標準出力へスタックトレースを出し、callback(None) を呼ぶ
- macOS では fork 優先（割り込み伝播のため）。無理なら spawn にフォールバック
"""

import time
import traceback
import multiprocessing as mp
from multiprocessing.queues import Queue
from typing import Callable, Any, List, Tuple, Optional


def _job_wrapper(func: Callable, args: tuple, idx: int, out_q: Queue):
    """ワーカ側：func(*args) を実行して結果を out_q に送る"""
    try:
        res = func(*args)
        out_q.put((idx, res, False, None))
    except Exception:
        tb = traceback.format_exc()
        out_q.put((idx, None, True, tb))


class Group:
    def __init__(self):
        # 各要素: (func, args, callback)
        self._jobs: List[Tuple[Callable, tuple, Optional[Callable[[Any], None]]]] = []

    def add_job(self, func: Callable, args: tuple, callback: Optional[Callable[[Any], None]] = None):
        self._jobs.append((func, args, callback))

    def run_jobs(self, num_cores: int):
        """最大 num_cores 並列で全ジョブを実行する"""
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")

        out_q: Queue = ctx.Queue()
        procs: List[mp.Process] = []
        total = len(self._jobs)
        next_job = 0
        active = 0
        finished = 0

        def start_one(jidx: int):
            func, args, _ = self._jobs[jidx]
            p = ctx.Process(target=_job_wrapper, args=(func, args, jidx, out_q), daemon=True)
            p.start()
            return p

        try:
            # 立ち上げ
            initial = min(num_cores, total)
            for _ in range(initial):
                p = start_one(next_job)
                procs.append(p)
                next_job += 1
                active += 1

            # ループ
            while finished < total:
                try:
                    idx, res, is_err, tb = out_q.get(timeout=0.1)
                except Exception:
                    # 追加起動
                    while active < num_cores and next_job < total:
                        p = start_one(next_job)
                        procs.append(p)
                        next_job += 1
                        active += 1
                    continue

                _, _, cb = self._jobs[idx]
                if is_err:
                    print(f"[mp_group] ERROR in job {idx}\n{tb}")
                    if cb is not None:
                        try:
                            cb(None)
                        except Exception:
                            pass
                else:
                    if cb is not None:
                        try:
                            cb(res)
                        except Exception:
                            pass

                finished += 1
                active -= 1

                # 補充起動
                while active < num_cores and next_job < total:
                    p = start_one(next_job)
                    procs.append(p)
                    next_job += 1
                    active += 1

                time.sleep(0.01)

        except KeyboardInterrupt:
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join(timeout=1.0)
            raise
        finally:
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join(timeout=1.0)