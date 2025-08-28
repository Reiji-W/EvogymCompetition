from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
from server.trainer.ga.base import Individual

class BaseCrossover(ABC):
    @abstractmethod
    def __call__(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """親2体 -> 子2体"""
        ...

class NoCrossover(BaseCrossover):
    """交叉を行わずコピーするだけ。"""
    def __call__(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        c1 = Individual(p1.body.copy(), p1.connections.copy(), p1.label, p1.controller_params)
        c2 = Individual(p2.body.copy(), p2.connections.copy(), p2.label, p2.controller_params)
        return c1, c2