#!/usr/bin/env bash
if [ $# -lt 2 ]
then
    echo "Usage: bash $0 LOG_OUT GPUS"
    exit
fi
#export CUDA_VISIBLE_DEVICES=0,1
LOG_OUT=$1
GPUS=$2
#GPUS=2
#LOG_OUT=./ckpts_global_criss_D2Tcross_kiba/kiba_dist.out
PORT=5555

if [ ${GPUS} == 1 ]
then
    nohup python train_global_criss_D2Tcross_dist.py -b 8 --lr 2e-4 --dataset benchmark_kiba 2>&1 >> $LOG_OUT &
else
    nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 --master_port=$PORT \
    train_global_criss_D2Tcross_dist.py --local_world_size $GPUS --distributed \
    -b 8 --lr 2e-4 --dataset benchmark_kiba 2>&1 >> $LOG_OUT &
fi