#! /bin/bash
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="visualglm-6b"
MODEL_ARGS="--max_source_length 64 \
    --max_target_length 256 \
    --lora_rank 10 \
    --layer_range 0 14 \
    --pre_seq_len 4"

# OPTIONS_SAT="SAT_HOME=$1" #"SAT_HOME=/raid/dm/sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

train_data="./fewshot-data/dataset.json"
eval_data="./fewshot-data/dataset.json"


gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 300 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --valid-data ${eval_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --save-interval 300 \
       --eval-interval 10000 \
       --save "./checkpoints" \
       --split 1 \
       --eval-iters 10 \
       --eval-batch-size 8 \
       --zero-stage 1 \
       --lr 0.0001 \
       --batch-size 1 \
       --gradient-accumulation-steps 4 \
       --skip-init \
       --fp16 \
       --use_qlora
"

              

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --include localhost:0 --hostfile ${HOST_FILE_PATH} finetune_visualglm.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
