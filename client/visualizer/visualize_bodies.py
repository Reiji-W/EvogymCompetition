#!/usr/bin/env python3
# client/visualizer/visualize_bodies.py
# client/mnt/<exp>/generation_<gen>/structure/*.npz を読み込み、
# 指定世代の全個体（全ランク）を EvoGym 配色でタイル表示（上下反転なし）

import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

MNT_ROOT = os.path.join("client", "mnt")

# ---------- ユーティリティ ----------

def natural_key(s: str):
    """ '12.npz', '3.npz' のような名前を数値順にするためのキー """
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def list_experiments():
    if not os.path.isdir(MNT_ROOT):
        raise FileNotFoundError(f"マウントディレクトリが見つかりません: {MNT_ROOT}")
    return sorted([d for d in os.listdir(MNT_ROOT)
                   if os.path.isdir(os.path.join(MNT_ROOT, d))])

def list_generations(exp_name: str):
    exp_dir = os.path.join(MNT_ROOT, exp_name)
    gens = []
    for d in os.listdir(exp_dir):
        if d.startswith("generation_"):
            try:
                gens.append(int(d.split("_", 1)[1]))
            except Exception:
                pass
    return sorted(gens)

def evogym_cmap_and_norm():
    """EvoGym配色（0..4のカテゴリ想定）"""
    colors = [
        (1.0, 1.0, 1.0),       # 0: EMPTY   white
        (0.0, 0.0, 0.0),       # 1: RIGID   black
        (0.6, 0.6, 0.6),       # 2: SOFT    gray
        (1.0, 0.55, 0.0),      # 3: H_ACT   orange
        (0.0, 0.8, 1.0),       # 4: V_ACT   cyan
    ]
    cmap = ListedColormap(colors, name="evogym")
    boundaries = np.arange(-0.5, len(colors) + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm

def load_bodies(exp_name: str, generation: int):
    struct_dir = os.path.join(MNT_ROOT, exp_name, f"generation_{generation}", "structure")
    if not os.path.isdir(struct_dir):
        raise FileNotFoundError(f"構造ディレクトリが見つかりません: {struct_dir}")

    files = [f for f in os.listdir(struct_dir) if f.endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"*.npz が見つかりません: {struct_dir}")

    files.sort(key=natural_key)

    bodies, labels, ctrl_params = [], [], []
    for f in files:
        path = os.path.join(struct_dir, f)
        data = np.load(path, allow_pickle=True)
        if "arr_0" not in data.files:
            continue
        body = data["arr_0"]
        bodies.append(body)
        label = os.path.splitext(f)[0]
        try:
            labels.append(int(label))
        except ValueError:
            labels.append(label)
        if "arr_2" in data.files:
            try:
                ctrl_params.append(tuple(np.array(data["arr_2"]).tolist()))
            except Exception:
                ctrl_params.append(None)
        else:
            ctrl_params.append(None)

    if not bodies:
        raise RuntimeError("ボディ配列の読み込みに失敗しました。")

    return bodies, labels, ctrl_params

# ---------- メイン ----------

def main():
    # 実験選択（インタラクティブ）
    exps = list_experiments()
    if not exps:
        print(f"{MNT_ROOT} に実験フォルダが見つかりません。")
        return
    print("Available experiments:")
    for i, e in enumerate(exps):
        print(f"  {i}: {e}")
    exp_in = input("\n実験を選択してください（番号または名前）: ").strip()
    if exp_in.isdigit():
        idx = int(exp_in)
        if not (0 <= idx < len(exps)):
            print("不正な番号です。終了します。")
            return
        exp_name = exps[idx]
    else:
        if exp_in not in exps:
            print("その実験フォルダは存在しません。終了します。")
            return
        exp_name = exp_in

    # 世代選択（インタラクティブ）
    gens = list_generations(exp_name)
    if not gens:
        print("generation_* が見つかりません。終了します。")
        return
    print("\nAvailable generations:")
    print(", ".join(map(str, gens[:50])) + (" ..." if len(gens) > 50 else ""))
    gen_in = input("世代番号を入力してください: ").strip()
    if not gen_in.isdigit():
        print("世代番号は整数で入力してください。終了します。")
        return
    generation = int(gen_in)
    if generation not in gens:
        print(f"generation_{generation} は存在しません。終了します。")
        return

    # 読み込み（全件）
    bodies, labels, ctrl_params = load_bodies(exp_name, generation)
    n = len(bodies)
    print(f"\n表示対象: {n} 件（全ランク）")

    # タイルレイアウト（自動調整）
    cols = max(1, min(10, int(math.ceil(math.sqrt(n)))))  # √n を基準に最大10列
    rows = int(math.ceil(n / cols))

    cmap, norm = evogym_cmap_and_norm()

    cell_w, cell_h = 2.0, 2.0
    fig_w, fig_h = max(6, cols * cell_w), max(4, rows * cell_h)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("off")
        if idx >= n:
            ax.imshow(np.zeros((2, 2)), cmap=cmap, norm=norm, interpolation="nearest")
            continue

        body = bodies[idx]
        lbl = labels[idx]
        ax.imshow(body, cmap=cmap, norm=norm, interpolation="nearest")  # 反転なし
        title = f"id {lbl} (rank {idx+1})"
        if ctrl_params[idx] is not None:
            f, a, p = ctrl_params[idx]
            title += f" | f={f:.3f}, a={a:.2f}, p={p:.2f}"
        ax.set_title(title, fontsize=9)

    # 凡例
    legend_items = [
        ("Empty", (1.0, 1.0, 1.0)),
        ("Rigid", (0.0, 0.0, 0.0)),
        ("Soft",  (0.6, 0.6, 0.6)),
        ("H Act", (1.0, 0.55, 0.0)),
        ("V Act", (0.0, 0.8, 1.0)),
    ]
    handles = [Patch(facecolor=col, edgecolor='k', label=name) for name, col in legend_items]
    fig.legend(handles=handles, loc="upper right")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

if __name__ == "__main__":
    main()