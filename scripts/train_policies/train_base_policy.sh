#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3
OBJECT=$4
TASK=$5

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=${TASK} headless=True seed=${SEED} \
train.algo=PPO \
task.env.object="${OBJECT}" \
train.ppo.proprio_adapt=False \
train.ppo.output_name="${CACHE}" \
${EXTRA_ARGS}