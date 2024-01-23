#!/bin/bash
EXP_NAME=$1

python train.py task=HANDEM headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
train.algo=HANDEM \
train.handem.output_name=test_logs/"${EXP_NAME}" \
checkpoint="${EXP_NAME}"