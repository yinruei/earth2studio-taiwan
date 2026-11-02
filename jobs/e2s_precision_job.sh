#!/bin/bash
# FCN3 BF16 autocast (Earth2Studio default) vs FP32: same node, seed and initial condition,
# ABBA order with two runs per precision, then the comparison table and figure.
# Submit from the deployment root:  pjsub jobs/e2s_precision_job.sh
# Edit the resource unit and queue below for your site. The job asks for a whole node but uses one GPU.
#PJM -N E2S_PREC
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
# torch.compile does not change FCN3 (makani leaves the network uncompiled); keep compile time out of the timings.
export TORCHDYNAMO_DISABLE=1

ARGS="--ic data/gfs_20241030T00_ic.nc --fcn3 models/fcn3 --corrdiff models/corrdiff_taiwan --nsteps 8 --samples 4"
i=0
for prec in bf16 fp32 fp32 bf16; do
    i=$((i+1))
    ${PY} -u scripts/run_fcn3_corrdiff.py ${ARGS} --fcn3-precision ${prec} --output results/${TAG}_${i}_${prec} || exit 30
done
plot_python scripts/compare_precision.py results/${TAG}_[1-4]_* --output results/${TAG}_precision || exit 32

nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv
echo "==== JOB END $(date '+%F %T') ===="
