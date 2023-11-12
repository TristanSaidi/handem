#!/bin/bash
GPUS=$1
EXP_NAME=$2
OBJECT=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"
mkdir outputs/trees/${EXP_NAME}

CUDA_VISIBLE_DEVICES=${GPUS} \
python zero_actions.py task=RRT headless=True \
task.env.numEnvs=16384 \
task.env.object="${OBJECT}" \
task.env.episodeLength=40 \
task.name=RRT \
task.RRT.save_file="${EXP_NAME}" \
${EXTRA_ARGS}