# Shared runtime setup for the PJM jobs; sourced after cd to the deployment root.
if [ ! -f site.env ]; then
    echo "site.env not found in $(pwd): copy site.env.example and edit it" >&2
    exit 2
fi
source ./site.env
ROOT=$(pwd)
PY=${ROOT}/venv/bin/python
PLOT_PY=${PLOT_PYTHON:-${PY}}
TAG=${pjtag:-run_${PJM_JOBID:-local}}

# torch-harmonics was built with gcc 13, so its libstdc++ must take precedence over the system one.
export LD_LIBRARY_PATH=${TOOLCHAIN}/lib:${LD_LIBRARY_PATH}
# GPU nodes ship /bin/gcc without cc1; Triton/Inductor JIT needs an explicit, complete compiler.
export CC=${TOOLCHAIN}/bin/x86_64-conda-linux-gnu-gcc CXX=${TOOLCHAIN}/bin/x86_64-conda-linux-gnu-g++
export PATH=${TOOLCHAIN}/bin:${PATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MPLBACKEND=Agg
export EARTH2STUDIO_CACHE=${ROOT}/cache HF_HUB_OFFLINE=1
# A fresh compile cache per job keeps first-step (compile) timings comparable between jobs.
export TORCHINDUCTOR_CACHE_DIR=${ROOT}/cache/inductor_${TAG} TRITON_CACHE_DIR=${ROOT}/cache/triton_${TAG}
mkdir -p log results

plot_python() {
    LD_LIBRARY_PATH=${PLOT_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH}} PYTHONWARNINGS=ignore ${PLOT_PY} "$@"
}

echo "==== JOB START $(date '+%F %T') host=$(hostname) tag=${TAG} ===="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
${PY} -c "import torch, disco_helpers, attention_helpers; print('optimized kernels: disco', disco_helpers.optimized_kernels_is_available(), 'attention', attention_helpers.optimized_kernels_is_available())"
${CC} --version | head -1
