## Env Builder / Design Tool

`client/env_builder/evogym-design-tool` は `evogym` 本体を import するため、実行する Python 環境（`venv`）に `evogym` が必要です。

### セットアップ（venv 推奨）

```powershell
python -m venv client\env_builder\.venv
client\env_builder\.venv\Scripts\Activate.ps1
python -m pip install -r client\env_builder\requirements_env_builder.txt
```

### 起動

```powershell
python client\env_builder\evogym-design-tool\src\main.py
```

## JSON の受け渡し（サーバに反映）

- Design Tool の出力は `client/env_builder/evogym-design-tool/exported/` に保存されます。
- サーバ側は `server/custom_env/active/` に **JSON 1 件だけ** がある状態を要求します（入れ替えは `server/custom_env/inbox/activate.sh` を使用）。

リモートサーバに送る場合は `client/config/remote.yaml` を用意して、以下を使います。

```powershell
copy client\config\remote.yaml.example client\config\remote.yaml
.\client\scripts\send_world_json.ps1 client\env_builder\evogym-design-tool\exported\world.json
```
