#!/bin/bash
EXP_NAME=$1
TASK=$2

python train.py task=${TASK} headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
train.algo=HANDEM \
train.handem.output_name=test_logs/"${EXP_NAME}" \
checkpoint="${EXP_NAME}"