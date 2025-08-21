#!/usr/bin/env bash

# scaffolding.sh
# 必要なディレクトリと空ファイルを生成する
# server
mkdir -p server/scripts
mkdir -p server/trainer/utils
mkdir -p server/custom_env/worlds
mkdir -p server/saved_data

touch server/requirements.txt
touch server/scripts/run_ga.sh
touch server/scripts/tail_logs.sh
touch server/trainer/__init__.py
touch server/trainer/ren_onlyGA.py
touch server/trainer/utils/__init__.py
touch server/trainer/utils/mp_group.py
touch server/trainer/utils/algo_utils.py
touch server/custom_env/__init__.py
touch server/custom_env/my_walker_env.py
touch server/custom_env/worlds/.gitkeep
touch server/saved_data/.gitkeep

# client
mkdir -p client/config
mkdir -p client/scripts
mkdir -p client/visualizer
mkdir -p client/env_builder/worlds
mkdir -p client/env_builder/tools
mkdir -p client/mnt    # マウントポイント

touch client/requirements.txt
touch client/config/remote.yaml
touch client/scripts/mount_saved_data.sh
touch client/scripts/umount_saved_data.sh
touch client/scripts/send_world_json.sh
touch client/scripts/check_remote.sh
touch client/visualizer/__init__.py
touch client/visualizer/visualize_bodies.py
touch client/visualizer/visualize_rollout.py
touch client/visualizer/export_robot_video.py
touch client/env_builder/README.md
touch client/env_builder/worlds/.gitkeep

# .github
mkdir -p .github/workflows
touch .github/workflows/ci.yml