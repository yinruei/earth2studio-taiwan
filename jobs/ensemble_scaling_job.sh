#!/bin/bash
# Strong scaling of a 16-member FCN3 -> CorrDiff Taiwan ensemble on one node: 1, 2 and 4 GPUs (one worker
# process per GPU, members split round-robin), plus two 4-GPU variants: NUMA-bound workers, and model
# weights staged to /dev/shm. Order: n4 first, so the most contended load is not helped by a warm cache.
# Submit from the deployment root:  pjsub jobs/ensemble_scaling_job.sh
# Edit the resource unit and queue below for your site.
#PJM -N E2S_ENS
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
export TORCHDYNAMO_DISABLE=1          # keep per-process compile time out of the scaling numbers
MEMBERS=${MEMBERS:-16}
OUT=results/${TAG}_ensemble
mkdir -p ${OUT}
nvidia-smi topo -m > ${OUT}/topology.txt

# GPU index -> NUMA node with direct PCIe attachment (from `nvidia-smi topo -m` on the tested node)
GPU_NUMA=(${GPU_NUMA:-0 0 1 1})

run_config() {   # run_config <name> <n_gpus> <mode: default|numa|shm>
    local name=$1 ngpu=$2 mode=$3 fcn3=models/fcn3 corrdiff=models/corrdiff_taiwan
    mkdir -p ${OUT}/${name}
    local launch staged
    launch=$(date +%s.%N)
    staged=${launch}         # no staging step unless mode=shm
    if [ "${mode}" = shm ]; then
        local shm=/dev/shm/e2s_models_${TAG}
        mkdir -p ${shm} && cp -r models/fcn3 models/corrdiff_taiwan ${shm}/ || exit 40
        fcn3=${shm}/fcn3; corrdiff=${shm}/corrdiff_taiwan
        staged=$(date +%s.%N)
    fi
    for ((g = 0; g < ngpu; g++)); do
        local members=$(seq ${g} ${ngpu} $((MEMBERS - 1)) | tr '\n' ' ')
        local prefix="" threads=${OMP_NUM_THREADS}
        if [ "${mode}" = numa ]; then
            prefix="numactl --cpunodebind=${GPU_NUMA[$g]} --membind=${GPU_NUMA[$g]}"
            threads=4                  # two GPUs share each GPU-attached 8-core NUMA node
        fi
        CUDA_VISIBLE_DEVICES=${g} OMP_NUM_THREADS=${threads} ${prefix} ${PY} -u scripts/ensemble_worker.py \
            --ic data/gfs_20241030T00_ic.nc --fcn3 ${fcn3} --corrdiff ${corrdiff} --members ${members} \
            --save-dir ${OUT}/${name}/fields --output ${OUT}/${name}/gpu${g}.json > ${OUT}/${name}/gpu${g}.log 2>&1 &
    done
    wait
    local done_t=$(date +%s.%N)
    [ "${mode}" = shm ] && rm -rf /dev/shm/e2s_models_${TAG}
    echo "{\"name\": \"${name}\", \"n_gpus\": ${ngpu}, \"mode\": \"${mode}\", \"launch\": ${launch}, \"staged\": ${staged}, \"done\": ${done_t}}" \
        > ${OUT}/${name}/wall.json
    echo "== ${name} done $(date '+%T')"
    grep -l Traceback ${OUT}/${name}/gpu*.log && exit 41
}

run_config n4 4 default
run_config n2 2 default
run_config n1 1 default
run_config n4_numa 4 numa
run_config n4_shm 4 shm
plot_python scripts/plot_ensemble_scaling.py ${OUT} --output ${OUT} || exit 42

nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv
echo "==== JOB END $(date '+%F %T') ===="
