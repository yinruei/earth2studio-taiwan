#!/bin/bash
# Run on an internet-connected workstation (tested from Git Bash on Windows with Python 3.11).
# Produces everything the air-gapped cluster needs under ${OUT}; copy it over with scp/tar.
# Keep ${OUT} short on Windows: long paths break pip and git checkouts.
set -euo pipefail
OUT=${OUT:-$TEMP/e2s}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "${OUT}"/{wheels,src,models/fcn3/training_checkpoints,models/corrdiff_taiwan,gcc13_pkgs,gfs}
cd "${OUT}"
PLAT="--platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 --platform manylinux_2_17_x86_64
      --platform manylinux2014_x86_64 --platform linux_x86_64 --python-version 3.12 --implementation cp
      --only-binary=:all: --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu126"

# 1. Python wheels for Linux / CPython 3.12.
#    antlr4-python3-runtime 4.9.3 (hydra-core dependency) ships only an sdist: build a wheel locally first.
python -m pip wheel "antlr4-python3-runtime==4.9.3" --no-deps -w wheels
python -m pip download -r "${HERE}/requirements-e2s.txt" -d wheels --find-links wheels ${PLAT}
python -m pip download earth2studio==0.20.0 --no-deps -d wheels
#    pip evaluates `platform_system == "Linux"` markers against the *local* OS, so on Windows it
#    silently skips torch's CUDA runtime wheels (nvidia-*-cu12, triton). Read them from the wheel.
LINUX_ONLY=$(python - <<'EOF'
import glob, re, zipfile
z = zipfile.ZipFile(glob.glob('wheels/torch-2.9.1*.whl')[0])
meta = z.read([n for n in z.namelist() if n.endswith('.dist-info/METADATA')][0]).decode()
print(' '.join(l[15:].split(';')[0].replace(' ', '') for l in meta.splitlines()
               if l.startswith('Requires-Dist:') and 'platform_system == "Linux"' in l))
EOF
)
python -m pip download ${LINUX_ONLY} -d wheels --find-links wheels ${PLAT}
#    warp-lang is pinned to 1.12.0 in requirements-e2s.txt (physicsnemo 2.0.0 needs wp.context).

# 2. Git-pinned sources that Earth2Studio 0.20.0 installs from GitHub (see its [tool.uv.sources]).
curl -fL -o src/makani-b38fcb2.tar.gz \
  https://github.com/NVIDIA/makani/archive/b38fcb2799d7dbc146fa60459f3f9823394a8bf1.tar.gz
curl -fL -o src/torch-harmonics-a632ca7.tar.gz \
  https://github.com/NVIDIA/torch-harmonics/archive/a632ca748a12bd9f74dbc1e00653317810991f74.tar.gz

# 3. Model weights (public, no NGC key needed).
HF=https://huggingface.co/nvidia/fourcastnet3/resolve/76ef0c60237e458b33196ba027134e27f3fc4538
for f in config.json global_means.npy global_stds.npy land_mask.nc maxs.npy metadata.json mins.npy \
         orography.nc README.md training_checkpoints/best_ckpt_mp0.tar; do
  curl -fL --retry 3 -o "models/fcn3/${f}" "${HF}/${f}"
done
curl -fL --retry 3 -o models/corrdiff_taiwan/corrdiff_inference_package.zip \
  https://api.ngc.nvidia.com/v2/models/nvidia/modulus/corrdiff_inference_package/versions/1/files/corrdiff_inference_package.zip

# 4. conda-forge gcc 13 + sysroot 2.28 (the cluster's gcc 8.5 cannot build torch 2.9 extensions).
#    gcc13_explicit.txt was produced with:
#      CONDA_SUBDIR=linux-64 conda create -n probe --dry-run --json --override-channels -c conda-forge \
#        "gxx_linux-64=13" "gcc_linux-64=13" "sysroot_linux-64=2.28"
#    Each line is file://@PKG_DIR@/<package>#<md5>; download the same package names here.
grep '^file://' "${HERE}/gcc13_explicit.txt" | sed 's#.*/##; s/#.*//' | while read -r pkg; do
  curl -fsL -o "gcc13_pkgs/${pkg}" "https://conda.anaconda.org/conda-forge/linux-64/${pkg}" ||
    curl -fL -o "gcc13_pkgs/${pkg}" "https://conda.anaconda.org/conda-forge/noarch/${pkg}"
done

# 5. Initial condition: only the 72 FCN3 input messages from one GFS analysis (~57 MiB).
python "${HERE}/fetch_gfs_ic.py" --wheel wheels/earth2studio-0.20.0-py3-none-any.whl \
  --date 20241030 --cycle 00 --output gfs
sha256sum models/fcn3/training_checkpoints/best_ckpt_mp0.tar models/corrdiff_taiwan/*.zip gfs/*.grib2
du -sh wheels src models gcc13_pkgs gfs
