from __future__ import annotations
import math, random
import numpy as np
from typing import Optional
from server.trainer.utils.algo_utils import Structure

class Individual(Structure):
    def __init__(
        self,
        body: np.ndarray,
        connections: np.ndarray,
        label: int,
        controller_params: Optional[tuple[float, float, float]] = None,
    ):
        super().__init__(body, connections, label)

        if controller_params is None:
            # 生成時点で float32 に丸めて保持
            controller_params = (
                np.float32(random.uniform(0.01, 0.07)),        # frequency
                np.float32(random.uniform(0.5,  1.5)),         # amplitude
                np.float32(random.uniform(0.0,  2*math.pi)),   # phase
            )

        # 渡されても最終的に float32 を保証
        self.controller_params = tuple(np.asarray(controller_params, dtype=np.float32).tolist())