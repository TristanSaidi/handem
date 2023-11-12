#!/bin/bash
EXP_NAME=$1
OBJECT=$2
TASK=$3

python train.py task=${TASK} headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
task.env.object="${OBJECT}" \
train.algo=HANDEM \
train.handem.output_name=test_logs/"${EXP_NAME}" \
checkpoint="${EXP_NAME}"