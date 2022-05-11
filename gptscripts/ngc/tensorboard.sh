#!/bin/bash

IMAGE=nvidia/pytorch:22.03-py3
NAME=ml-model.notamodel-tb-gpt-512sl
INSTANCE=cpu.x86.tiny

PORT=6006
WORKSPACE=/gpt2-zh
CKPNAME=oscar-512sl
CHECKPOINT_PATH=${WORKSPACE}/ckp/${CKPNAME}
LOG_PATH=${CHECKPOINT_PATH}
RESUTL=/result

COMMAND="cd ${LOG_PATH}; tensorboard --logdir log --port ${PORT};"
ngc batch run --ace nv-us-west-2 --org nvidian --team sae \
              --name ${NAME} \
              --image ${IMAGE} \
              --instance ${INSTANCE} \
              --commandline "${COMMAND}" \
              --preempt RUNONCE \
              --result ${RESUTL} \
              --workspace ZydrZx5GQmSSIYDaJmulLw:${WORKSPACE}:RO \
              --port ${PORT}
