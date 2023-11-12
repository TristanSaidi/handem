#!/bin/bash
GPUS=$1
TREE_PATH=$2
SAVE_FILE=$3
OBJECT=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python zero_actions.py task=RRT headless=False \
task.env.episodeLength=100000 \
task.env.numEnvs=1 \
task.env.object="${OBJECT}" \
task.name=ExtractGrasps \
task.RRT.samples_per_iter=1 \
task.extract_grasps.save_file="${SAVE_FILE}" \
task.extract_grasps.tree_path="${TREE_PATH}" \
${EXTRA_ARGS}