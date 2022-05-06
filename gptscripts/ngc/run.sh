#!/bin/bash
# +------------------------+-----------------------+-----------------------+--------------+-----------------------+--------+---------------------+-------+-----------+
# | Id                     | Name                  | Description           | ACE          | Creator Username      | Shared | Created Date        | Owned | Size      |
# | ZydrZx5GQmSSIYDaJmulLw | gpt2-zh               |                       | nv-us-west-2 | Ray Zhang             | No     | 2022-04-22 07:51:06 | Yes   | 0 B       |

# +-------------------+------------+-------------------+-------------------+--------------+--------+------------+-----------+--------------+-------+---------+
# | Id                | Integer Id | Name              | Description       | ACE          | Shared | Size       | Status    | Created Date | Owned | Pre-pop |
# | PT7e0qA9R1icfYUty | 99309      | oscar_zh          |                   | nv-us-west-2 | No     | 78.97 GB   | COMPLETED | 2022-04-21   | Yes   | No      |
# | 0Mqh86FUT_2zqhnxbO5-4Q  | 99729      | oscarcorpus     |                 | nv-us-west-2 | No     | 55.72 GB   | COMPLETED | 2022-04-29   | Yes   | No      |

# NAME=ml-model.notamodel-wiki-zh-process.exempt-tc-gpu
# NAME=ml-model.notamodel-gpt2-zh-oscar-debug.exempt-tc-gpu
NAME=ml-model.gpt2-zh-oscar
# INSTANCE=dgxa100.40g.8.norm
INSTANCE=dgx1v.16g.8.norm
# INSTANCE=cpu.x86.tiny
IMAGE=nvidia/pytorch:22.03-py3


WORKSPACE=/gpt2-zh
DATASETID=99729    # oscar
DATADIR=/data
MEGATRON=${WORKSPACE}/Megatron-LM
DATA_PATH=${WORKSPACE}/oscar/oscar_text_document
DATAKEY=text
VOCAB_PATH=${MEGATRON}/vocab/jq.zh.v2.vocab
EXE=${MEGATRON}/pretrain_gpt_zh.py
CHECKPOINT_PATH=${WORKSPACE}/ckp/gpt-oscar-v100
LOG_PATH=${CHECKPOINT_PATH}/log
LOGFILE=${CHECKPOINT_PATH}/cmdlog.log


# BATCH_PER_GPU=64
# BATCH=512
LR=0.00015
MAX_SEQ_LEN=1024
ITERS=800000
SAVE_INTERVAL=10000
EVAL_INTERVAL=1000


GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=7000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

OUTPUT_ARGS="--log-interval 100 \
              --tensorboard-dir ${LOG_PATH} \
              --tensorboard-log-interval 50 \
              --no-log-loss-scale-to-tensorboard \
              --save-interval ${SAVE_INTERVAL} \
              --eval-interval ${EVAL_INTERVAL} \
              --eval-iters 10 \
              --activations-checkpoint-method uniform \
              --save ${CHECKPOINT_PATH} \
              --load ${CHECKPOINT_PATH}"

# rm -rf ${WORKSPACE}/oscar; \
# cp -rf ${DATADIR}/* ${WORKSPACE}/oscar; \
COMMAND="cd ${MEGATRON}; \
       pip install zhconv; \
       mkdir  ${WORKSPACE}/oscar; \
       mkdir ${CHECKPOINT_PATH};
       mkdir ${CHECKPOINT_PATH}/results; \
       python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
       ${EXE} \
       ${OUTPUT_ARGS} \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 32 \
       --global-batch-size 256 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters ${ITERS} \
       --lr-decay-iters 320000 \
       --data-path ${DATA_PATH} \
       --vocab-file ${VOCAB_PATH} \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr ${LR} \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --fp16 | tee $LOGFILE"

ngc batch run --ace nv-us-west-2 --org nvidian --team sae \
              --name ${NAME} \
              --image ${IMAGE} \
              --instance ${INSTANCE} \
              --commandline "${COMMAND}" \
              --result ${CHECKPOINT_PATH}/results \
              --preempt RUNONCE \
              --datasetid ${DATASETID}:${DATADIR} \
              --workspace ZydrZx5GQmSSIYDaJmulLw:${WORKSPACE}:RW
