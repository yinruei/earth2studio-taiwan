#!/bin/bash
# One offline forecast: GFS 2024-10-30 00 UTC (Typhoon Kong-rey) -> FCN3 48 h -> CorrDiff Taiwan, 4 samples.
# Submit from the deployment root:  pjsub jobs/e2s_job_large.sh
# Edit the resource unit and queue below for your site. The job asks for a whole node but uses one GPU.
#PJM -N E2S_TW
#PJM -L rscunit=YOUR_GPU_RSCUNIT
#PJM -L rscgrp=YOUR_GPU_QUEUE
#PJM -L vnode=1(core=32)
#PJM -L gpu=4
#PJM -P "vn-policy=abs-unpack,exec-policy=simplex"
#PJM -L elapse=01:00:00
#PJM -o log/e2s.%j.out
#PJM -e log/e2s.%j.err

cd "${PJM_O_WORKDIR:-$(pwd)}" || exit 10
source jobs/common.sh

${PY} -u scripts/run_fcn3_corrdiff.py --ic data/gfs_20241030T00_ic.nc --fcn3 models/fcn3 \
    --corrdiff models/corrdiff_taiwan --nsteps 8 --samples 4 --output results/${TAG} || exit 30

nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv
echo "==== JOB END $(date '+%F %T') ===="
