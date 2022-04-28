#!/bin/bash

IMAGE=nvidia/pytorch:22.03-py3
NAME=ml-model.notmodal-oscar-clue-preprocess
INSTANCE=cpu.x86.tiny

WORKERS=8
WORKSPACE=/mnt/gpt2-zh
MEGATRON=${WORKSPACE}/Megatron-LM
INPUT=${MEGATRON}/data/pl_oscar_zh.json
DATA_PREFIX=${MEGATRON}/data/oscar
VOCAB=${MEGATRON}/vocab/jq.zh.v2.vocab   # added a few chinese symbols
KEYS=text

EXE=${MEGATRON}/tools/preprocess_data.py  # For Chinese
COMMAND="pwd; cd ${MEGATRON}; pwd; \
python ${EXE} \
       --input '${INPUT}' \
       --output-prefix ${DATA_PREFIX} \
       --vocab ${VOCAB} \
       --json-keys ${KEYS} \
       --dataset-impl mmap \
       --workers ${WORKERS} \
       --tokenizer-type GPT2BertWordPieceTokenizer \
       --append-eod ;
       sleep 3600"

#     --datasetid ${DATASETID}:${DATASET} \

ngc batch run --name ${NAME} --image ${IMAGE} --instance ${INSTANCE} --commandline "${COMMAND}" --result ${MEGATRON}/result/results \
    --preempt RUNONCE \
    --total-runtime 36000s \
    --ace nv-us-west-2 --org nvidian --team sae \
    --workspace qG7q67EMSma_OzVg98v7-A:${WORKSPACE}:RW
