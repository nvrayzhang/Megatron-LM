#!/bin/bash

IMAGE=pytorch2203
MEGATRON=/home/nvidia/Projects/Megatron-LM

INPUT=/data/converted/debug.json

VOCAB=${MEGATRON}/vocab/RoBERTa-wwm-ext-large.vocab

KEYS=text
DATA_PREFIX=${MEGATRON}/data/oscar

WORKERS=8


# EXE=tools/zh/preprocess_data_zh.py   # For Chinese
EXE=${MEGATRON}/tools/preprocess_data.py   # For Chinese
docker exec ${IMAGE} bash -c "cd ${MEGATRON}; \
python ${EXE} \
       --input '${INPUT}' \
       --output-prefix ${DATA_PREFIX} \
       --vocab ${VOCAB} \
       --json-keys ${KEYS} \
       --dataset-impl mmap \
       --workers ${WORKERS} \
       --tokenizer-type GPT2BertWordPieceTokenizer \
       --append-eod
       "
