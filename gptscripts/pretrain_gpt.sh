#! /bin/bash
# Runs the "345M" parameter model
IMAGE=pytorch2203



MEGATRON=/home/nvidia/Projects/Megatron-LM
VOCAB=${MEGATRON}/vocab/RoBERTa-wwm-ext-large.vocab
DATA_PATH=${MEGATRON}/data/oscar_text_document

EXE=${MEGATRON}/pretrain_gpt.py

CHECKPOINT_PATH=/tmp/gpt
mkdir -p ${CHECKPOINT_PATH}
LOG_PATH=${CHECKPOINT_PATH}/log



MASTER_ADDR=localhost
MASTER_PORT=7000
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

docker exec ${IMAGE} bash -c "cd ${MEGATRON}; \
echo MASTER_ADDR=${MASTER_ADDR}; \
echo MASTER_PORT=${MASTER_PORT}; \
torchrun  ${EXE} \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save ${CHECKPOINT_PATH} \
       --load ${CHECKPOINT_PATH} \
       --data-path ${DATA_PATH} \
       --vocab-file ${VOCAB} \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --activations-checkpoint-method uniform \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 "
