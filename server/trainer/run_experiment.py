# server/trainer/run_experiment.py
from __future__ import annotations
import os, sys, shutil, argparse, math, random
from typing import List
import numpy as np
from evogym import sample_robot, hashable

from server.trainer.utils.mp_group import Group
from server.trainer.ga.base import Individual
from server.trainer.ga.engine import resolve_env, copy_active_assets           # ← evaluate_structure を消す
from server.trainer.ga.evaluator import evaluate_structure                     # ← ここからインポート
from server.trainer.ga.registry import get_mutation, get_crossover, get_selection


def save_generation(home_path: str, generation: int, structures: List[Individual]) -> None:
    gen_dir = os.path.join(home_path, f"generation_{generation}")
    struct_dir = os.path.join(gen_dir, "structure")
    os.makedirs(struct_dir, exist_ok=True)
    with open(os.path.join(gen_dir, "output.txt"), "w") as fout:
        for s in structures:
            np.savez(
                os.path.join(struct_dir, f"{s.label}.npz"),
                s.body,
                s.connections,
                np.array(s.controller_params, dtype=np.float32),
            )
            f_str = ",".join(f"{v:.4f}" for v in s.controller_params)
            fout.write(f"{s.label}\t{s.fitness:.4f}\t{f_str}\n")


def run_experiment(
    exp_name: str,
    env_name: str | None,
    pop_size: int,
    structure_shape: tuple[int, int],
    max_evaluations: int,
    num_cores: int,
    max_steps: int,
    max_episode_steps: int | None = None,
    mutation_name: str = "default",
    crossover_name: str = "none",
    selection_name: str = "truncation",
    use_custom_env: bool = True,
) -> None:
    env_id, is_custom = resolve_env(env_name, max_episode_steps, force_custom=use_custom_env)
    home_path = os.path.join("server/saved_data", exp_name)
    if os.path.exists(home_path):
        shutil.rmtree(home_path)
    os.makedirs(home_path, exist_ok=True)
    if is_custom:
        copy_active_assets(home_path, env_id)

    # 実験メタデータを保存（ベース/カスタム共通で ENV 名を記録）
    metadata_path = os.path.join(home_path, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write("ALGO: GA\n")
        f.write(f"ENV: {env_id}\n")
        f.write(f"POP_SIZE: {pop_size}\n")
        f.write(f"STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n")
        f.write(f"MAX_EVALUATIONS: {max_evaluations}\n")
        f.write(f"MAX_STEPS: {max_steps}\n")
        if max_episode_steps is not None:
            f.write(f"MAX_EPISODE_STEPS: {max_episode_steps}\n")
        try:
            import evogym, gymnasium, numpy as _np  # type: ignore

            f.write(
                f"VERSIONS: evogym={getattr(evogym, '__version__', 'unknown')}, "
                f"gymnasium={getattr(gymnasium, '__version__', 'unknown')}, "
                f"numpy={_np.__version__}\n"
            )
        except Exception:
            pass

    mutation = get_mutation(mutation_name)
    crossover = get_crossover(crossover_name)
    selection = get_selection(selection_name)

    # 初期集団
    structures: List[Individual] = []
    seen_hashes = set()
    num_evals, gen = 0, 0
    for i in range(pop_size):
        body, connections = sample_robot(structure_shape)
        while hashable(body) in seen_hashes:
            body, connections = sample_robot(structure_shape)
        structures.append(Individual(body, connections, i))
        seen_hashes.add(hashable(body))
        num_evals += 1

    # 進化ループ
    while num_evals <= max_evaluations:
        print(f"Generation {gen} | evals {num_evals}/{max_evaluations}")
        group = Group()
        for s in structures:
            group.add_job(
                evaluate_structure,
                (s.body, s.connections, s.controller_params, env_id, max_steps),
                callback=s.set_reward,
            )
        group.run_jobs(num_cores)

        structures.sort(key=lambda x: x.fitness, reverse=True)
        save_generation(home_path, gen, structures)

        if num_evals >= max_evaluations:
            break

        survivors, lam = selection(structures, pop_size, num_evals, max_evaluations)
        children: List[Individual] = []
        next_label = len(survivors)

        while len(children) < lam and num_evals < max_evaluations:
            if random.random() < 0.5:
                p1, p2 = random.sample(survivors, 2)
                c1, c2 = crossover(p1, p2)
                for child in (c1, c2):
                    if len(children) >= lam or num_evals >= max_evaluations:
                        break
                    child.label = next_label
                    children.append(child)
                    seen_hashes.add(hashable(child.body))
                    next_label += 1
                    num_evals += 1
            else:
                parent = random.choice(survivors)
                child = mutation(parent, next_label)
                if child is None:
                    continue
                if hashable(child.body) in seen_hashes:
                    continue
                children.append(child)
                seen_hashes.add(hashable(child.body))
                next_label += 1
                num_evals += 1

        structures = survivors + children
        gen += 1

    print("GA complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA for EvoGym")
    parser.add_argument("--exp_name", type=str, default="default_experiment")
    parser.add_argument("--env_name", type=str, default="")
    parser.add_argument("--pop_size", type=int, default=120)
    parser.add_argument("--structure_shape", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--max_evaluations", type=int, default=1200)
    parser.add_argument("--num_cores", type=int, default=12)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--mutation", type=str, default="default")
    parser.add_argument("--crossover", type=str, default="none")
    parser.add_argument("--selection", type=str, default="truncation")
    parser.add_argument(
        "--custom_env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="カスタム環境を使う（デフォルト True）。ベース環境に戻す場合は --no-custom_env。",
    )
    args = parser.parse_args()

    run_experiment(
        exp_name=args.exp_name,
        env_name=args.env_name,
        pop_size=args.pop_size,
        structure_shape=tuple(args.structure_shape),
        max_evaluations=args.max_evaluations,
        num_cores=args.num_cores,
        max_steps=args.max_steps,
        max_episode_steps=args.max_episode_steps,
        mutation_name=args.mutation,
        crossover_name=args.crossover,
        selection_name=args.selection,
        use_custom_env=args.custom_env,
    )
