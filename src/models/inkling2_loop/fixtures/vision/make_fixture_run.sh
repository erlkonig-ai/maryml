#!/bin/bash
# CPU-only, offline, read-only inputs. The container is the memory cap.
set -euo pipefail
L=/home/liora/lanes/mini-vision-fixture-20261123
T=/home/liora/lanes/mini-native-layer0-20261122/transformers-046bdb8/src   # git 046bdb8, clean
D=/home/liora/lanes/mini-native-layer0-20261122/pydeps                     # huggingface_hub 1.33.0, tokenizers 0.23.3
M=/media/liora/PILES/scratch-mini-20261122/Inkling-2-Mini                  # revision b85c3cfa
IMG=vllm/vllm-openai@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
exec nice -n 10 docker run --rm --network none --memory 12g --cpus 4 \
  -e CUDA_VISIBLE_DEVICES= -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/pydeps:/tf -e PYTHONDONTWRITEBYTECODE=1 -e USER=liora -e HOME=/tmp -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor --tmpfs /tmp:size=1g \
  -v $T:/tf:ro -v $D:/pydeps:ro -v $M:/model:ro -v $L/scripts:/scripts:ro \
  -v $L/out:/out -v $L/fixture:/fixture --user $(id -u):$(id -g) \
  --entrypoint python3 $IMG /scripts/make_fixture.py /model /out /fixture
