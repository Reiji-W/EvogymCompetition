# Visualizer（結果の可視化）

可視化は以下の 2 つを使います。

- `client/visualizer/visualize_bodies.py`：世代内の全個体をタイル表示
- `client/visualizer/visualize_rollout.py`：選んだ個体をロールアウトして描画

## 仮想環境（推奨）

```powershell
python -m venv client\visualizer\.venv
client\visualizer\.venv\Scripts\Activate.ps1
python -m pip install -r client\requirements_client.txt
```

## データ取得（サーバ saved_data を同期）

`client/mnt/` に `server/saved_data/` の中身を同期してから可視化します。

- 設定：`client/config/remote.yaml`（`client/config/remote.yaml.example` をコピーして編集）
- 同期：
  - Windows: `.\client\scripts\mount_saved_data.ps1`
  - Bash: `bash client/scripts/mount_saved_data.sh`

## 可視化

```powershell
python client\visualizer\visualize_bodies.py
python client\visualizer\visualize_rollout.py
```

`visualize_rollout.py` は `metadata.txt` と実験ディレクトリ直下の同梱 JSON（存在する場合）を使って、学習時と同じ環境ID/ワールドで再現する設計です。

