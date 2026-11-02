#!/bin/bash
# PhysicsNeMo AFNO data-parallel training on one node: generate the FCN3 regional training set (once), then
# weak-scaling benchmarks on 1/2/4 GPUs in FP32 and BF16, an Nsight Systems capture of 4-GPU BF16 steps
# (NCCL all-reduce), and a longer 4-GPU training run with validation loss.
# Submit from the deployment root:  pjsub jobs/train_scaling_job.sh
# Edit the resource unit and queue below for your site.
#PJM -N E2S_TRAIN
#PJM -L rscunit=YOUR_GPU_RSCUNIT
#PJM -L rscgrp=YOUR_GPU_QUEUE
#PJM -L vnode=1(core=32)
#PJM -L gpu=4
#PJM -P "vn-policy=abs-unpack,exec-policy=simplex"
#PJM -L elapse=02:00:00
#PJM -o log/e2s.%j.out
#PJM -e log/e2s.%j.err

cd "${PJM_O_WORKDIR:-$(pwd)}" || exit 10
source jobs/common.sh
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=4
DATA=data/afno_fcn3_taiwan
OUT=results/${TAG}_train
mkdir -p ${OUT}

if [ ! -f ${DATA}/meta.json ]; then
    echo "== generate training data $(date '+%T')"
    for g in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=${g} ${PY} -u scripts/gen_training_data.py --ic data/gfs_20241030T00_ic.nc \
            --fcn3 models/fcn3 --members $(seq ${g} 4 31 | tr '\n' ' ') --nsteps 16 --output ${DATA} \
            > ${OUT}/gen_gpu${g}.log 2>&1 &
    done
    wait
    grep -l Traceback ${OUT}/gen_gpu*.log && exit 20
    [ -f ${DATA}/meta.json ] || exit 21
fi
ls ${DATA} | wc -l

launch() {   # launch <n_gpus> <args...>
    local n=$1; shift
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) ${PY} -m torch.distributed.run --standalone --nproc_per_node=${n} \
        scripts/train_afno_ddp.py --data ${DATA} "$@"
}

for prec in fp32 bf16; do
    for n in 1 2 4; do
        echo "== bench n=${n} ${prec} $(date '+%T')"
        launch ${n} --precision ${prec} --output ${OUT}/bench_n${n}_${prec}.json > ${OUT}/bench_n${n}_${prec}.log 2>&1 || exit 30
        tail -1 ${OUT}/bench_n${n}_${prec}.log
    done
done

echo "== nsys 4-GPU bf16 $(date '+%T')"
NSYS=${NSYS:-nsys}
CUDA_VISIBLE_DEVICES=0,1,2,3 ${NSYS} profile -t cuda,nvtx -s none --cpuctxsw=none --capture-range=cudaProfilerApi \
    --capture-range-end=stop --force-overwrite=true -o ${OUT}/ddp_n4_bf16 \
    ${PY} -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train_afno_ddp.py --data ${DATA} \
    --precision bf16 --steps 20 --profile --output ${OUT}/profile_n4_bf16.json > ${OUT}/profile.log 2>&1 || exit 31
for rep in ${OUT}/ddp_n4_bf16*.nsys-rep; do
    ${NSYS} stats --report cuda_gpu_kern_sum --format csv --force-export=true --output ${rep%.nsys-rep} ${rep} > /dev/null
done

echo "== training run n=4 bf16 $(date '+%T')"
launch 4 --precision bf16 --steps 3000 --warmup 1 --val-every 100 --output ${OUT}/train_n4_bf16.json \
    > ${OUT}/train_n4_bf16.log 2>&1 || exit 32
tail -2 ${OUT}/train_n4_bf16.log

nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv
echo "==== JOB END $(date '+%F %T') ===="
