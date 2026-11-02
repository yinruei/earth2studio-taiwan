"""Ensemble worker: FCN3 -> CorrDiff Taiwan for a list of members on one GPU, with wall-clock timestamps.

Members differ only by seed (FCN3's stochastic noise and CorrDiff's sampler), all from the same analysis.
Several workers run side by side, one per GPU; the job script splits members round-robin. Each member's
outputs are hashed (SHA-256 of every CorrDiff input and output tensor), so a member computed on 1 GPU can
be compared bit for bit with the same member computed while 4 GPUs share the node.
"""
import time

PROCESS_START = time.time()

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch

_extractall = zipfile.ZipFile.extractall


def extract_if_needed(self, path=None, members=None, pwd=None):
    """CorrDiffTaiwan.load_model unzips its package on every load. Concurrent workers would overwrite the
    same files while others read them, so skip extraction when every file is already present and complete."""
    root = Path(path or '.')
    complete = all((root / i.filename).is_dir() if i.is_dir() else
                   (root / i.filename).is_file() and (root / i.filename).stat().st_size == i.file_size
                   for i in self.infolist())
    if not complete:
        _extractall(self, path, members, pwd)


zipfile.ZipFile.extractall = extract_if_needed


def synced(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ic', type=Path, required=True)
    p.add_argument('--fcn3', type=Path, required=True)
    p.add_argument('--corrdiff', type=Path, required=True)
    p.add_argument('--members', type=int, nargs='+', required=True)
    p.add_argument('--nsteps', type=int, default=8)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--base-seed', type=int, default=20241030)
    p.add_argument('--save-lead', type=int, default=30, help='save the CorrDiff sample mean at this lead (hours)')
    p.add_argument('--save-dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from earth2studio.data import DataArrayFile, fetch_data
    from earth2studio.models.auto import Package
    from earth2studio.models.dx import CorrDiffTaiwan
    from earth2studio.models.px import FCN3
    from earth2studio.utils.coords import map_coords
    imports_done = time.time()

    device = torch.device('cuda')
    fcn3, fcn3_load = synced(lambda: FCN3.load_model(Package(str(a.fcn3))).to(device))
    corrdiff, corrdiff_load = synced(lambda: CorrDiffTaiwan.load_model(Package(str(a.corrdiff)), device=device).to(device))
    data = DataArrayFile(str(a.ic))
    ic = fcn3.input_coords()
    (x0, coords0), fetch = synced(lambda: fetch_data(data, time=data.da['time'].values[:1], variable=ic['variable'],
                                                       lead_time=ic['lead_time'], device=device))
    x0, coords0 = map_coords(x0, coords0, ic)
    compute_start = time.time()

    members = []
    for m in a.members:
        seed = a.base_seed + m
        fcn3.set_rng(seed=seed)
        corrdiff.seed, corrdiff.number_of_samples = seed, a.samples
        t_start, digest = time.time(), hashlib.sha256()
        fcn3_sec = corrdiff_sec = copy_sec = 0.0
        model = fcn3.create_iterator(x0.clone(), coords0)
        for _ in range(a.nsteps + 1):
            (x, coords), dt = synced(lambda: next(model))
            fcn3_sec += dt
            xd, cdc = map_coords(x, coords, corrdiff.input_coords())
            (y, _), dt = synced(lambda: corrdiff(xd, cdc))
            corrdiff_sec += dt
            t = time.perf_counter()
            yc = y.cpu().numpy()
            digest.update(xd.cpu().numpy().tobytes())
            digest.update(yc.tobytes())
            lead = int(coords['lead_time'][-1] / np.timedelta64(1, 'h'))
            if a.save_dir is not None and lead == a.save_lead:
                a.save_dir.mkdir(parents=True, exist_ok=True)
                np.save(a.save_dir / f'member_{m:02d}_lead{lead:02d}h_mean.npy',
                        yc.reshape(a.samples, -1, *yc.shape[-2:]).mean(axis=0))
            copy_sec += time.perf_counter() - t
        members.append(dict(member=m, seed=seed, start=t_start, end=time.time(), fcn3_sec=fcn3_sec,
                            corrdiff_sec=corrdiff_sec, copy_hash_sec=copy_sec, sha256=digest.hexdigest()))
        print(json.dumps({k: v for k, v in members[-1].items() if k != 'sha256'}), flush=True)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(
        gpu=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        omp_threads=os.environ.get('OMP_NUM_THREADS'), cpu_affinity=sorted(os.sched_getaffinity(0)),
        process_start=PROCESS_START, imports_done=imports_done, fcn3_load_sec=fcn3_load,
        corrdiff_load_sec=corrdiff_load, fetch_sec=fetch, compute_start=compute_start, end=time.time(),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, nsteps=a.nsteps, samples=a.samples,
        members=members), indent=2))


if __name__ == '__main__':
    main()
