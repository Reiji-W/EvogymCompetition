# server/trainer/ga/operators/selections.py
from __future__ import annotations
from abc import ABC, abstractmethod
import math
from typing import List, Tuple
from server.trainer.ga.base import Individual
from server.trainer.utils.algo_utils import get_percent_survival_evals


class BaseSelection(ABC):
    @abstractmethod
    def __call__(
        self,
        structures: List[Individual],
        pop_size: int,
        num_evals: int,
        max_evaluations: int,
    ) -> Tuple[List[Individual], int]:
        """
        構成個体 `structures` は fitness で降順ソート済みである前提。
        戻り値: (survivors, lambda_count)
          - survivors: 次世代に生き残る個体（ラベルは 0..μ-1 に振り直す）
          - lambda_count: 生成すべき子個体数（λ）
        """
        ...


class TruncationSelection(BaseSelection):
    """
    しきい選択（上位 μ を残す）。μ は進捗率に応じて増加:
      μ = ceil(pop_size * get_percent_survival_evals(num_evals, max_evaluations))
      λ = pop_size - μ
    """
    def __call__(
        self,
        structures: List[Individual],
        pop_size: int,
        num_evals: int,
        max_evaluations: int,
    ) -> Tuple[List[Individual], int]:
        # 進捗率に応じて μ を決定（下限 2）
        pct = get_percent_survival_evals(num_evals, max_evaluations)
        mu = max(2, math.ceil(pop_size * pct))
        lam = max(1, pop_size - mu)

        survivors = structures[:mu]
        # ラベル振り直し（既存コード踏襲）
        for idx, s in enumerate(survivors):
            s.is_survivor = True
            s.prev_gen_label = s.label
            s.label = idx

        return survivors, lam


__all__ = ["BaseSelection", "TruncationSelection"]