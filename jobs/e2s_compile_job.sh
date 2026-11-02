#!/bin/bash
# torch.compile vs eager on the same node, seed and initial condition (compiled first, then eager).
# Submit from the deployment root:  pjsub jobs/e2s_compile_job.sh
# Edit the resource unit and queue below for your site. The job asks for a whole node but uses one GPU.
#PJM -N E2S_CMP
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
# Show whether the node's own gcc can compile (on the tested cluster it has no cc1).
ls /bin/gcc $(dirname $(readlink -f /bin/gcc))/../libexec/gcc/x86_64-redhat-linux/*/cc1 2>&1 | head -3

ARGS="--ic data/gfs_20241030T00_ic.nc --fcn3 models/fcn3 --corrdiff models/corrdiff_taiwan --nsteps 8 --samples 4"
${PY} -u scripts/run_fcn3_corrdiff.py ${ARGS} --output results/${TAG}_compiled || exit 30
TORCHDYNAMO_DISABLE=1 ${PY} -u scripts/run_fcn3_corrdiff.py ${ARGS} --output results/${TAG}_eager || exit 31
PYTHONWARNINGS=ignore ${PY} scripts/compare_runs.py results/${TAG}_compiled results/${TAG}_eager \
    | tee results/${TAG}_comparison.txt

nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv
echo "==== JOB END $(date '+%F %T') ===="
