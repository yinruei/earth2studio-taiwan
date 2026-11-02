#!/bin/bash
# CorrDiff Taiwan inference benchmark (precision, diffusion steps, sample batching, torch.compile),
# then an Nsight Systems capture of two default calls with NVTX ranges per network.
# Submit from the deployment root:  pjsub jobs/corrdiff_bench_job.sh
# Edit the resource unit and queue below for your site. The job asks for a whole node but uses one GPU.
#PJM -N E2S_CDBENCH
#PJM -L rscunit=YOUR_GPU_RSCUNIT
#PJM -L rscgrp=YOUR_GPU_QUEUE
#PJM -L vnode=1(core=32)
#PJM -L gpu=4
#PJM -P "vn-policy=abs-unpack,exec-policy=simplex"
#PJM -L elapse=01:30:00
#PJM -o log/e2s.%j.out
#PJM -e log/e2s.%j.err

cd "${PJM_O_WORKDIR:-$(pwd)}" || exit 10
source jobs/common.sh
OUT=results/${TAG}_corrdiff
ARGS="--ic data/gfs_20241030T00_ic.nc --fcn3 models/fcn3 --corrdiff models/corrdiff_taiwan --lead 30"

${PY} -u scripts/corrdiff_bench.py ${ARGS} --output ${OUT} || exit 30

NSYS=${NSYS:-nsys}
${NSYS} profile -t cuda,nvtx -s none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true -o ${OUT}/base_profile \
    ${PY} -u scripts/corrdiff_bench.py ${ARGS} --profile --output ${OUT} || exit 31
${NSYS} stats --report cuda_gpu_kern_sum,nvtx_sum --format csv --force-export=true \
    --output ${OUT}/base_profile ${OUT}/base_profile.nsys-rep > /dev/null || exit 32
ls -la ${OUT}

nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv
echo "==== JOB END $(date '+%F %T') ===="
