#!/bin/bash

IMAGE=pytorch2203
stage=0
MEGATRON=/home/nvidia/Projects/Megatron-LM
INPUT=/data/converted/train.json

VOCAB=vocab/clue.vocab

if [ ${stage} -eq 0 ]; then
 #EXE=tools/analyze_vocab.py
  EXE=tools/zh/find_unknowns.py
  docker exec ${IMAGE} bash -c "cd ${MEGATRON};\
  set -f;
  python ${EXE} \
         --input ${INPUT} \
         --vocab ${VOCAB} \
         --workers 16 \
         --tokenizer-type BertWordPieceLowerCase | tee last_run.log"
  exit 0
fi
