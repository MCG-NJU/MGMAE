#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='YOUR_PATH/work_dir/vit_b_k400_mgmae_800e'
DATA_PATH='YOUR_PATH/data/k400/k400_train.csv'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        --async \
        ${SRUN_ARGS} \
        python -u run_mgmae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type mgmae \
        --mask_ratio 0.9 \
        --init_mask_map mix_gauss \
        --model pretrain_videomae_base_patch16_224 \
        --get_flow raft \
        --flow_model '/your/path/model_zoo/raft-small-clean.pth' \
        --flow_iter 6 \
        --base_frame middle \
        --warp_type backward \
        --hole_filling consist \
        --decoder_depth 4 \
        --batch_size 24 \
        --num_sample 4 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_workers 10 \
        --lr 1e-3 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 20 \
        --save_ckpt_freq 20 \
        --epochs 200 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        ${PY_ARGS}
