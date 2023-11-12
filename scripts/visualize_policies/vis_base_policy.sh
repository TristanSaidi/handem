#!/bin/bash
CACHE=$1
OBJECT=$2
TASK=$3

python train.py task=${TASK} headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
task.env.object="${OBJECT}" \
train.algo=PPO \
task.env.randomization.randomizationCurriculum=False \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeObjScale=False \
task.env.randomization.randomizeCalibrationError=False \
task.env.randomization.randomizeJointFriction=False \
task.env.randomization.randomizeVelocityLimit=False \
task.env.randomization.randomizeTorqueLimit=False \
task.env.default.mass=0.12 \
task.env.default.friction=1 \
task.env.default.scale=1 \
task.env.default.calibrationError=0.00 \
task.env.default.stiffness=30 \
task.env.default.damping=5 \
task.env.default.velocityLimit=1 \
task.env.default.effortLimit=4 \
task.env.default.jointFriction=0.0 \
train.ppo.output_name=test_logs/"${CACHE}" \
checkpoint="${CACHE}"