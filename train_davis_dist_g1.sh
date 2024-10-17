GPUS=1
OUT_LOG=./ckpts_global_criss_D2Tcross_kiba/kiba_dist.out
export CUDA_VISIBLE_DEVICES=0
PORT=5555

if [ ${GPUS} == 1 ]; then
    python train_global_criss_D2Tcross_dist.py \
    -b 8 --lr 6e-4 \
    --dataset benchmark_kiba 2>&1 | tee -a $OUT_LOG
else
    nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 --master_port=$PORT \
    train_global_criss_D2Tcross_dist.py --local_world_size $GPUS --distributed \
    -b 8 --lr 6e-4 --dataset benchmark_kiba 2>&1 | tee -a $OUT_LOG &
fi
