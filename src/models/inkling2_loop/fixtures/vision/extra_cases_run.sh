#!/bin/bash
# CPU-only, offline, processor only. Same inputs as mini-vision-fixture-20261123/scripts/run.sh.
set -euo pipefail
L=/home/liora/lanes/mini-vision-preprocess-20261123/ref
T=/home/liora/lanes/mini-native-layer0-20261122/transformers-046bdb8/src
D=/home/liora/lanes/mini-native-layer0-20261122/pydeps
M=/media/liora/PILES/scratch-mini-20261122/Inkling-2-Mini
IMG=vllm/vllm-openai@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
exec nice -n 10 docker run --rm --network none --memory 4g --cpus 2 \
  -e CUDA_VISIBLE_DEVICES= -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/pydeps:/tf -e PYTHONDONTWRITEBYTECODE=1 -e USER=liora -e HOME=/tmp --tmpfs /tmp:size=256m \
  -v $T:/tf:ro -v $D:/pydeps:ro -v $M:/model:ro -v $L/scripts:/scripts:ro -v ${OUT:-$L/out}:/out --user $(id -u):$(id -g) \
  --entrypoint python3 $IMG /scripts/extra_cases.py /model /out
