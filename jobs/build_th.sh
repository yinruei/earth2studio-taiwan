#!/bin/bash
# Build torch-harmonics (commit a632ca7, pinned by Earth2Studio 0.20.0) with its CUDA kernels for A100 (sm_80)
# using conda-forge gcc 13 + nvcc, then install makani (commit b38fcb2). Run on a login node from the
# deployment root after venv/ exists:  bash jobs/build_th.sh
set -eo pipefail
source ./site.env
ROOT=$(pwd)
export CUDA_HOME=${CUDA_TOOLKIT}
export PATH=${CUDA_HOME}/bin:${TOOLCHAIN}/bin:${PATH}
export CC=${TOOLCHAIN}/bin/x86_64-conda-linux-gnu-gcc CXX=${TOOLCHAIN}/bin/x86_64-conda-linux-gnu-g++
# torch's CUDA headers include cusparse.h; NVIDIA HPC SDK keeps it outside CUDA_HOME.
export CPATH=${CUDA_MATH_INCLUDE}:${CPATH}
export NVCC_PREPEND_FLAGS="-ccbin ${CXX} -I${CUDA_MATH_INCLUDE}"
# The login node has no GPU, so force the CUDA build and name the target architecture.
export FORCE_CUDA_EXTENSION=1 TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=8
export LD_LIBRARY_PATH=${TOOLCHAIN}/lib:${LD_LIBRARY_PATH}

mkdir -p "${ROOT}/log"
cd "${ROOT}/src"
for archive in makani-b38fcb2.tar.gz torch-harmonics-a632ca7.tar.gz; do
    [ -d "$(tar -tzf ${archive} | head -1 | cut -d/ -f1)" ] || tar -xzf ${archive}
done
echo "== torch-harmonics $(date '+%T')"
${ROOT}/venv/bin/python -m pip install --no-index --no-build-isolation --no-deps -v \
    ./torch-harmonics-a632ca748a12bd9f74dbc1e00653317810991f74 > "${ROOT}/log/build_torch_harmonics.log" 2>&1 \
    || { tail -30 "${ROOT}/log/build_torch_harmonics.log"; exit 1; }
echo "== makani $(date '+%T')"
${ROOT}/venv/bin/python -m pip install --no-index --no-build-isolation --no-deps \
    ./makani-b38fcb2799d7dbc146fa60459f3f9823394a8bf1 2>&1 | tail -2

echo "== check $(date '+%T')"
cd "${ROOT}"
${ROOT}/venv/bin/python - <<'PY'
import torch, torch_harmonics, disco_helpers, attention_helpers, makani, earth2studio
print('torch', torch.__version__, 'torch-harmonics', torch_harmonics.__version__, 'earth2studio', earth2studio.__version__)
print('disco optimized kernels:', disco_helpers.optimized_kernels_is_available(),
      'attention:', attention_helpers.optimized_kernels_is_available())
PY
