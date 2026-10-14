set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
dataset="${DATASET:-thumos}"
first_split="${FIRST_SPLIT:-0}"
last_split="${LAST_SPLIT:-0}"
for ((split=first_split; split<=last_split; split++)); do
    python src/run.py data="$dataset" model="$dataset" model.setting=50 model.split="$split" "$@"
done
