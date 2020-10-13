#!/usr/bin/env bash
set -eux
echo "Usage: build_container.sh <amygda_repo_URL> <commit_id_or_branch>"
sudo docker build --build-arg amygda_repo_URL="$1" --build-arg commit_id_or_branch="$2" . -t leandroishilima/amygda:"$2"
