# Server（学習/進化計算）

## 仮想環境（推奨）

```powershell
python -m venv server/.venv
server/.venv/Scripts/Activate.ps1
python -m pip install -r server/requirements_server.txt
```

## カスタム環境（World JSON）

このリポジトリでは、カスタム環境は `server/custom_env` 配下で管理します。

- `server/custom_env/active/`：学習に使う **JSON を 1 件だけ** 置く場所
- `server/custom_env/inbox/`：受け取り用（ここに置いてから `activate.sh` で active に反映）
- `server/custom_env/archive/`：過去の active を退避
- `server/custom_env/worlds/`：`MyJsonWorld-N.json` として **env_id に紐付けて固定**（再現性用）

### 反映（inbox → active）

サーバ上で、`server/custom_env/inbox` に JSON を 1 件だけ置いた状態で：

```bash
bash server/custom_env/inbox/activate.sh
```

### 代替：環境変数で JSON を直指定

`EVOGYM_ACTIVE_JSON_OVERRIDE` に JSON の絶対パスを指定すると、`active/` のチェックをスキップしてそれを使用します。

## GA の実行（run_experiment.py）

実行コマンド（例）：

```powershell
python server/trainer/run_experiment.py --exp_name exp001 --custom_env --pop_size 120 --max_evaluations 1200 --num_cores 12 --max_steps 1000
```

出力は `server/saved_data/<exp_name>/` に保存されます（同名フォルダがある場合は **削除して作り直します**）。

### 主要パラメータ

`server/trainer/run_experiment.py` の引数：

- `--exp_name`：保存先フォルダ名（`server/saved_data/<exp_name>`）
- `--custom_env / --no-custom_env`：カスタム環境を使うか（デフォルトは `--no-custom_env`）
- `--env_name`：環境ID（`--no-custom_env` のときに使用。例：`Walker-v0`）
- `--pop_size`：集団サイズ（μ+λ の合計）
- `--structure_shape W H`：ロボット形状のグリッドサイズ（例：`5 5`）
- `--max_evaluations`：評価回数上限（初期個体生成+子生成でカウント）
- `--num_cores`：並列評価のワーカ数
- `--max_steps`：1評価あたりの最大ステップ数
- `--max_episode_steps`：Gym 側の max_episode_steps（カスタム env 登録時に使用。未指定なら既定値）
- `--mutation`：突然変異オペレータ名（現状：`default`）
- `--crossover`：交叉オペレータ名（現状：`none`）
- `--selection`：選択オペレータ名（現状：`truncation`）

## いじる場所（設計）

サーバ側の改造ポイントは主にここです。

- `server/trainer/run_experiment.py`：GA の流れ（評価→選択→交叉/突然変異→保存）
- `server/trainer/ga/operators/`：オペレータ実装
  - `mutations.py`：突然変異
  - `crossovers.py`：交叉
  - `selections.py`：選択（μ と λ の決め方など）
- `server/trainer/ga/registry.py`：オペレータ名（`--mutation` 等）と実装の対応表

オペレータを追加する場合は、`operators/*.py` にクラスを追加し、`registry.py` の辞書に名前を登録してください。

