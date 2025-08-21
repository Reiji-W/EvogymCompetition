# otani-lab-competition

このリポジトリは、大谷研究室内で行う進化的ロボティクスコンペティションのためのコード基盤です。
サーバで進化計算（GA）を実行し、生成されたロボットを学生がローカルで可視化・分析する運用を想定しています。

## ディレクトリ構成

```
server/                 # サーバサイド（進化計算実行環境）
  ├── scripts/          # 実行・ログ取得用スクリプト
  ├── trainer/          # アルゴリズム実装
  ├── custom_env/       # 環境定義
  └── saved_data/       # 結果保存

client/                 # クライアントサイド（可視化・環境作成）
  ├── config/           # 接続先設定(remote.yamlなど)
  ├── scripts/          # 転送・マウント系スクリプト
  ├── visualizer/       # 可視化コード
  ├── env_builder/      # 環境作成用ツール
  └── mnt/              # サーバのsaved_dataをマウント

.github/workflows/      # CI/CDワークフロー
```

## 利用方法

### 1. サーバサイド
- `server/scripts/run_ga.sh` を用いて進化計算を実行します。
- `server/saved_data/` に成果物（ログ・解ファイルなど）が保存されます。

### 2. クライアントサイド
- `client/config/remote.yaml` にサーバのアドレスやパスを設定します。
- `client/scripts/mount_saved_data.sh` を使ってサーバの結果をマウントします。
- `client/visualizer/` のスクリプトを用いて結果を可視化します。

### 3. 環境作成
- `client/env_builder/` に環境定義JSONを配置し、`client/scripts/send_world_json.sh` でサーバに転送します。

## ライセンス
このリポジトリは教育目的で使用されます。
