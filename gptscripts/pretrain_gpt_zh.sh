#! /bin/bash
# Runs the "345M" parameter model
IMAGE=pytorch2203

# ROOT DIR OF MEGATRON
MEGATRON=/home/nvidia/Projects/Megatron-LM
EXE=${MEGATRON}/pretrain_gpt_zh.py
VOCAB_PATH=${MEGATRON}/vocab/jq.zh.v2.vocab
DATA_PATH=${MEGATRON}/data/oscar_text_document
CHECKPOINT_PATH=/data/checkpoint
LOG_PATH=${CHECKPOINT_PATH}/log
LOGFILE=${CHECKPOINT_PATH}/cmdlog.log

# DISTRIBUTED SETTINGS
MASTER_ADDR=localhost
MASTER_PORT=9980
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# TRAINING SETTINGS
LR=0.00015
ITERS=50000
SAVE_INTERVAL=10000
EVAL_INTERVAL=1000

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

OUTPUT_ARGS="--log-interval 100 \
              --tensorboard-dir ${LOG_PATH} \
              --tensorboard-log-interval 50 \
              --no-log-loss-scale-to-tensorboard \
              --log-interval 100 \
              --save-interval ${SAVE_INTERVAL} \
              --eval-interval ${EVAL_INTERVAL} \
              --eval-iters 10 \
              --activations-checkpoint-method uniform \
              --save ${CHECKPOINT_PATH} \
              --load ${CHECKPOINT_PATH}"

docker exec ${IMAGE} bash -c "cd ${MEGATRON}; mkdir ${CHECKPOINT_PATH}; mkdir ${LOG_PATH}; \
python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
       ${EXE} \
       ${OUTPUT_ARGS} \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 64 \
       --global-batch-size 128 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters ${ITERS} \
       --lr-decay-iters 320000 \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_PATH \
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
