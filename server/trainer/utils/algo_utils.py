# server/trainer/utils/algo_utils.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvoGym 依存の軽量ユーティリティ群
- Structure: fitness = reward の最小実装（set_reward で更新）
- TerminationCondition: 反復回数で停止判定
- mutate: ボディ配列をランダム置換し、妥当性 (connected & has_actuator) を満たせば
          (child_body, full_connectivity) を返す
- サバイバル率等のヘルパ関数は従来の式をそのまま踏襲
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import numpy as np
from evogym import (
    is_connected,
    has_actuator,
    get_full_connectivity,
    draw,
    get_uniform,
)


class Structure:
    """個体構造（最小版）。fitness は reward をそのまま用いる。"""

    def __init__(self, body: np.ndarray, connections: np.ndarray, label: int):
        self.body = body
        self.connections = connections

        self.reward: float = 0.0
        self.fitness: float = self.compute_fitness()

        self.is_survivor: bool = False
        self.prev_gen_label: int = 0
        self.label: int = label

    def compute_fitness(self) -> float:
        self.fitness = float(self.reward)
        return self.fitness

    def set_reward(self, reward: Optional[float]):
        """並列評価失敗時など None が来る可能性を考慮"""
        self.reward = float(reward) if reward is not None else -1e9
        self.compute_fitness()

    def __str__(self) -> str:
        return (
            f"\n\nStructure:\n{self.body}\n"
            f"F: {self.fitness}\tR: {self.reward}\tID: {self.label}"
        )

    def __repr__(self) -> str:
        return self.__str__()


class TerminationCondition:
    """最大反復数で停止判定するシンプルな条件。"""

    def __init__(self, max_iters: int):
        self.max_iters = int(max_iters)

    def __call__(self, iters: int) -> bool:
        return int(iters) >= self.max_iters

    def change_target(self, max_iters: int):
        self.max_iters = int(max_iters)


def mutate(child: np.ndarray, mutation_rate: float = 0.1, num_attempts: int = 10
           ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    ボディ child（整数カテゴリの 2D 配列）をランダム置換して妥当なロボットを返す。
    妥当性: is_connected(child) and has_actuator(child)
    成功: (child, get_full_connectivity(child)) を返す
    失敗: None
    """
    # セル種別の一様分布（EvoGym 定義）。空セル（0）が出やすいように重みを上げる
    pd = get_uniform(5)   # 0..4: EMPTY, RIGID, SOFT, ACT_H, ACT_V
    pd[0] = 0.6           # 空セルを出やすくして探索を広げる

    child = child.copy()

    for _ in range(num_attempts):
        # 各セルに mutation_rate の確率で新しいカテゴリをサンプル
        for i in range(child.shape[0]):
            for j in range(child.shape[1]):
                if draw([mutation_rate, 1 - mutation_rate]) == 0:
                    child[i, j] = draw(pd)

        if is_connected(child) and has_actuator(child):
            return child, get_full_connectivity(child)

    # 妥当解が見つからなかった
    return None


# ---- 選択・可視化補助 -------------------------------------------------

def get_percent_survival(gen: int, max_gen: int) -> float:
    """世代ベースの生存率（従来式をそのまま使用）"""
    low = 0.0
    high = 0.8
    if max_gen <= 1:
        return low
    return ((max_gen - gen - 1) / (max_gen - 1)) ** 1.5 * (high - low) + low


def total_robots_explored(pop_size: int, max_gen: int) -> int:
    """総探索個体数（従来式をそのまま）"""
    total = pop_size
    for i in range(1, max_gen):
        total += pop_size - max(2, math.ceil(pop_size * get_percent_survival(i - 1, max_gen)))
    return total


def total_robots_explored_breakpoints(pop_size: int, max_gen: int, max_evaluations: int) -> List[int]:
    """各世代における累積探索数のブレークポイント（上限付き）"""
    total = pop_size
    out = [total]
    for i in range(1, max_gen):
        total += pop_size - max(2, math.ceil(pop_size * get_percent_survival(i - 1, max_gen)))
        if total > max_evaluations:
            total = max_evaluations
        out.append(total)
    return out


def search_max_gen_target(pop_size: int, evaluations: int) -> int:
    """評価回数予算 evaluations を満たす最小の世代数を探索"""
    target = 0
    while total_robots_explored(pop_size, target) < evaluations:
        target += 1
    return target


def parse_range(str_inp: str, rbt_max: int) -> List[int]:
    """'1-3 7 10-12' のような入力を数値リストへ展開"""
    inp_with_spaces = ""
    out: List[int] = []

    for token in str_inp:
        if token == "-":
            inp_with_spaces += " - "
        else:
            inp_with_spaces += token

    tokens = inp_with_spaces.split()
    count = 0
    while count < len(tokens):
        if (count + 1) < len(tokens) and tokens[count].isnumeric() and tokens[count + 1] == "-":
            curr = tokens[count]
            last = rbt_max
            if (count + 2) < len(tokens) and tokens[count + 2].isnumeric():
                last = tokens[count + 2]
            for i in range(int(curr), int(last) + 1):
                out.append(i)
            count += 3
        else:
            if tokens[count].isnumeric():
                out.append(int(tokens[count]))
            count += 1
    return out


def pretty_print(list_org: List[int], max_name_length: int = 30) -> None:
    """簡易段組み出力（視覚確認用のユーティリティ）"""
    list_formatted: List[List[int]] = [[] for _ in range(len(list_org) // 4 + 1)]
    for i, val in enumerate(list_org):
        row = i % (len(list_org) // 4 + 1)
        list_formatted[row].append(val)

    print()
    for row in list_formatted:
        out = ""
        for el in row:
            out += str(el) + " " * (max_name_length - len(str(el)))
        print(out)


def get_percent_survival_evals(curr_eval: int, max_evals: int) -> float:
    """評価回数に基づく生存率（従来式を踏襲）"""
    low = 0.0
    high = 0.6
    if max_evals <= 1:
        return low
    return ((max_evals - curr_eval - 1) / (max_evals - 1)) * (high - low) + low


def total_robots_explored_breakpoints_evals(pop_size: int, max_evals: int) -> List[int]:
    """評価回数ベースでの累積探索ブレークポイント"""
    num_evals = pop_size
    out = [num_evals]
    while num_evals < max_evals:
        num_survivors = max(2, math.ceil(pop_size * get_percent_survival_evals(num_evals, max_evals)))
        new_robots = pop_size - num_survivors
        num_evals += new_robots
        if num_evals > max_evals:
            num_evals = max_evals
        out.append(num_evals)
    return out


if __name__ == "__main__":
    # 簡単な自己テスト（前と同じ出力傾向）
    pop_size = 25
    num_evals = pop_size
    max_evals = 750

    count = 1
    print(num_evals, num_evals, count)
    while num_evals < max_evals:
        num_survivors = max(2, math.ceil(pop_size * get_percent_survival_evals(num_evals, max_evals)))
        new_robots = pop_size - num_survivors
        num_evals += new_robots
        count += 1
        print(new_robots, num_evals, count)

    print(total_robots_explored_breakpoints_evals(pop_size, max_evals))