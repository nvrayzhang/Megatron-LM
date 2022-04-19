#!/bin/bash

IMAGE=pytorch2203
MEGATRON=/home/nvidia/Projects/Megatron-LM

INPUT=/data/converted/train.json

VOCAB=${MEGATRON}/vocab/clue.vocab

KEYS=content
DATA_PREFIX=${MEGATRON}/data/oscar

MAX_LEN=512
SEED=13
WORKERS=16
DEBUG=0

# EXE=tools/zh/preprocess_data_zh.py   # For Chinese
EXE=${MEGATRON}/tools/zh/preprocess_new2016zh.py   # For Chinese
docker exec ${IMAGE} bash -c "cd `pwd`; \
cd ${MEGATRON}; \
python ${EXE} \
       --input '${INPUT}' \
       --output-prefix ${DATA_PREFIX} \
       --vocab ${VOCAB} \
       --json-keys ${KEYS} \
       --max-sent-length ${MAX_LEN} \
       --debug ${DEBUG} \
       --seed ${SEED} \
       --dataset-impl mmap \
       --split-sentences \
       --workers ${WORKERS} \
       --tokenizer-type BertWordPieceLowerCase"
