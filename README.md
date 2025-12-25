# otani-lab-competition

大谷研コンペ向けの EvoGym ベース環境です。サーバ側で GA を回して `server/saved_data/` に保存し、クライアント側で同期して可視化します。

## ワークフロー

1. （Client）`env_builder` で World JSON を作る
2. （Client→Server）World JSON をサーバへ送って **active** に反映
3. （Server）`run_experiment.py` で進化計算（GA）を実行し `server/saved_data/<exp_name>/` に保存
4. （Client）`mount_saved_data` で `server/saved_data/` を `client/mnt/` に同期
5. （Client）`visualize_bodies.py` / `visualize_rollout.py` で可視化

## 仮想環境は分ける（衝突回避）

用途ごとに `venv` を分離してください（numpy / pillow / OpenGL 周りが衝突しやすい）。

- サーバ（学習/GA）：`server/.venv`（`server/requirements_server.txt`）
- env_builder（設計ツール）：`client/env_builder/.venv`（`client/env_builder/requirements_env_builder.txt`）
- visualizer（可視化）：`client/visualizer/.venv`（`client/requirements_client.txt`）

## ドキュメント

- Server（学習/パラメータ/改造ポイント）：`server/README.md`
- Env Builder（設計ツール）：`client/env_builder/README.md`
- Visualizer（可視化）：`client/visualizer/README.md`

## 改造ポイント（サーバ側の設計）

サーバ側はここをいじる前提です。

- `server/trainer/run_experiment.py`：GA の流れ（評価→選択→生成→保存）と引数
- `server/trainer/ga/operators/`：突然変異・交叉・選択
- `server/trainer/ga/registry.py`：`--mutation` / `--crossover` / `--selection` の名前解決
