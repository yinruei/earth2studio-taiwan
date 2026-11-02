#!/bin/bash
# Python start-up cost: import torch + the FCN3 and CorrDiff classes from the venv on the shared parallel
# file system, versus a venv unpacked from one tar file onto node-local disk.
# Prerequisite (login node, once):  tar -cf venv.tar venv
# Submit from the deployment root:  pjsub jobs/import_staging_job.sh
# Edit the resource unit and queue below for your site.
#PJM -N E2S_IMPORT
#PJM -L rscunit=YOUR_GPU_RSCUNIT
#PJM -L rscgrp=YOUR_GPU_QUEUE
#PJM -L vnode=1(core=32)
#PJM -L gpu=4
#PJM -P "vn-policy=abs-unpack,exec-policy=simplex"
#PJM -L elapse=00:40:00
#PJM -o log/e2s.%j.out
#PJM -e log/e2s.%j.err

cd "${PJM_O_WORKDIR:-$(pwd)}" || exit 10
# Not jobs/common.sh: its start-up check imports torch from the venv and would warm the cache being measured.
source ./site.env
export LD_LIBRARY_PATH=${TOOLCHAIN}/lib:${LD_LIBRARY_PATH} TORCHDYNAMO_DISABLE=1 PYTHONDONTWRITEBYTECODE=1
TAG=${pjtag:-run_${PJM_JOBID:-local}}
OUT=results/${TAG}_import
# Node-local disk. /dev/shm is often mounted noexec, which makes the CUDA shared libraries unloadable.
STAGE=${STAGE_DIR:-/tmp}/e2s_venv_${TAG}
mkdir -p ${OUT}
trap 'rm -rf ${STAGE}' EXIT
grep -E " (/dev/shm|/tmp) " /proc/mounts > ${OUT}/mounts.txt 2>&1
IMPORT='import torch; from earth2studio.models.px import FCN3; from earth2studio.models.dx import CorrDiffTaiwan
import earth2studio; print(earth2studio.__file__)'
echo "{\"host\": \"$(hostname)\", \"venv_files\": $(find venv -type f | wc -l), \"venv_tar_bytes\": $(stat -c %s venv.tar)}" \
    > ${OUT}/context.json

measure() {   # measure <label> <python> [parallel workers]
    local label=$1 py=$2 n=${3:-1} t0 t1 ok=true pids=()
    t0=$(date +%s.%N)
    for ((i = 0; i < n; i++)); do ${py} -c "${IMPORT}" > ${OUT}/${label}_${i}.out 2>&1 & pids+=($!); done
    for pid in "${pids[@]}"; do wait ${pid} || ok=false; done
    t1=$(date +%s.%N)
    echo "{\"label\": \"${label}\", \"workers\": ${n}, \"ok\": ${ok}, \"seconds\": $(awk "BEGIN{print ${t1} - ${t0}}"), \"module\": \"$(tail -1 ${OUT}/${label}_0.out)\"}" \
        | tee -a ${OUT}/import_times.jsonl
}

measure lustre_first venv/bin/python
measure lustre_second venv/bin/python
measure lustre_4_workers venv/bin/python 4

t0=$(date +%s.%N)
mkdir -p ${STAGE} && tar -xf venv.tar -C ${STAGE} || exit 20
t1=$(date +%s.%N)
echo "{\"label\": \"stage_tar_to_local\", \"workers\": 1, \"ok\": true, \"seconds\": $(awk "BEGIN{print ${t1} - ${t0}}"), \"module\": \"\"}" \
    | tee -a ${OUT}/import_times.jsonl

measure local_first ${STAGE}/venv/bin/python
measure local_second ${STAGE}/venv/bin/python
measure local_4_workers ${STAGE}/venv/bin/python 4
echo "==== JOB END $(date '+%F %T') ===="
