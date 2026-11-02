"""FCN3 global forecast -> CorrDiff Taiwan km-scale downscaling, offline, with stage timings.

Mirrors earth2studio.run.diagnostic (fetch -> map_coords -> prognostic iterator ->
map_coords -> diagnostic) but times each stage with CUDA synchronization and keeps
a Taiwan-region FCN3 subset plus every CorrDiff sample for later inspection.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import earth2studio.models.px.fcn3 as fcn3_module
import numpy as np
import torch
import xarray as xr
from earth2studio.data import DataArrayFile, fetch_data
from earth2studio.models.auto import Package
from earth2studio.models.dx import CorrDiffTaiwan
from earth2studio.models.px import FCN3
from earth2studio.utils.coords import map_coords

REGION = dict(lat=slice(32, 15), lon=slice(112, 130))
FCN3_KEEP = ['t2m', 'u10m', 'v10m', 'msl', 'tcwv', 'z500']


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ic', type=Path, required=True, help='DataArrayFile from make_ic.py')
    p.add_argument('--fcn3', type=Path, required=True)
    p.add_argument('--corrdiff', type=Path, required=True)
    p.add_argument('--nsteps', type=int, default=8, help='6-hour FCN3 steps')
    p.add_argument('--samples', type=int, default=4, help='CorrDiff samples per lead')
    p.add_argument('--seed', type=int, default=20241030)
    p.add_argument('--fcn3-precision', choices=['bf16', 'fp32'], default='bf16',
                   help='bf16 = Earth2Studio default autocast when the torch-harmonics CUDA extension is present')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    timings = dict(steps=[])

    fcn3, timings['fcn3_load_sec'] = timed(lambda: FCN3.load_model(Package(str(a.fcn3))))
    fcn3.set_rng(seed=a.seed)
    if not fcn3_module._cuda_extension_available:
        raise RuntimeError('torch-harmonics CUDA extension missing; precision comparison would also change kernels')
    if a.fcn3_precision == 'fp32':
        # FCN3._forward reads this module flag only to pick the autocast dtype; kernels stay the same.
        fcn3_module._cuda_extension_available = False
    corrdiff, timings['corrdiff_load_sec'] = timed(lambda: CorrDiffTaiwan.load_model(Package(str(a.corrdiff))))
    corrdiff.number_of_samples = a.samples
    corrdiff.seed = a.seed
    (fcn3, corrdiff), timings['to_device_sec'] = timed(lambda: (fcn3.to(device), corrdiff.to(device)))

    data = DataArrayFile(str(a.ic))
    init = data.da['time'].values[:1]
    prognostic_ic, diagnostic_ic = fcn3.input_coords(), corrdiff.input_coords()
    (x, coords), timings['fetch_sec'] = timed(lambda: fetch_data(
        data, time=init, variable=prognostic_ic['variable'], lead_time=prognostic_ic['lead_time'], device=device))
    x, coords = map_coords(x, coords, prognostic_ic)
    torch.cuda.reset_peak_memory_stats()

    model = fcn3.create_iterator(x, coords)
    fcn3_frames, cd_frames, leads = [], [], []
    for step in range(a.nsteps + 1):
        (x, coords), fcn3_sec = timed(lambda: next(model))
        lead = coords['lead_time'][-1]
        (xd, cdc), map_sec = timed(lambda: map_coords(x, coords, diagnostic_ic))
        (y, ycoords), cd_sec = timed(lambda: corrdiff(xd, cdc))
        io_start = time.perf_counter()
        sub = xr.DataArray(x.cpu().numpy(), dims=list(coords), coords={k: v for k, v in coords.items()})
        sub = sub.sel(variable=FCN3_KEEP, **REGION).squeeze(drop=True)
        fcn3_frames.append(sub)
        cd_frames.append(y.cpu().numpy().reshape(a.samples, len(ycoords['variable']), *y.shape[-2:]))
        leads.append(lead)
        io_sec = time.perf_counter() - io_start
        timings['steps'].append(dict(step=step, lead_hours=int(lead / np.timedelta64(1, 'h')), fcn3_sec=fcn3_sec,
                                     map_sec=map_sec, corrdiff_sec=cd_sec, host_copy_sec=io_sec))
        print(json.dumps(timings['steps'][-1]), flush=True)
    timings['peak_allocated_gib_during_rollout'] = torch.cuda.max_memory_allocated() / 2**30

    lead_hours = [int(l / np.timedelta64(1, 'h')) for l in leads]
    fcn3_ds = xr.concat(fcn3_frames, dim='lead_time').assign_coords(lead_time=lead_hours)
    fcn3_ds.attrs.update(init=str(init[0]), model='FCN3 (hf nvidia/fourcastnet3@76ef0c6)', seed=a.seed)
    fcn3_ds.to_netcdf(a.output / 'fcn3_taiwan_region.nc')
    lat2d, lon2d = np.asarray(ycoords['lat']), np.asarray(ycoords['lon'])
    cd = xr.DataArray(np.stack(cd_frames), dims=['lead_time', 'sample', 'variable', 'y', 'x'],
                      coords=dict(lead_time=lead_hours, sample=np.arange(a.samples),
                                  variable=list(ycoords['variable'])), name='corrdiff')
    if lat2d.ndim == 2:
        cd = cd.assign_coords(lat=(('y', 'x'), lat2d), lon=(('y', 'x'), lon2d))
    else:
        cd = cd.assign_coords(lat=('y', lat2d), lon=('x', lon2d))
    cd.attrs.update(init=str(init[0]), model='CorrDiffTaiwan (ngc nvidia/modulus/corrdiff_inference_package@1)',
                    seed=a.seed, samples=a.samples)
    cd.to_netcdf(a.output / 'corrdiff_taiwan.nc')

    steps = timings['steps'][1:]  # step 0 returns the initial condition
    timings['summary'] = dict(
        fcn3_step_median_sec=float(np.median([s['fcn3_sec'] for s in steps])),
        corrdiff_per_lead_median_sec=float(np.median([s['corrdiff_sec'] for s in timings['steps']])),
        corrdiff_per_sample_median_sec=float(np.median([s['corrdiff_sec'] for s in timings['steps']]) / a.samples),
        total_rollout_sec=float(sum(s['fcn3_sec'] + s['map_sec'] + s['corrdiff_sec'] + s['host_copy_sec'] for s in timings['steps'])))
    timings['environment'] = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                                  gpu=torch.cuda.get_device_name(0), nsteps=a.nsteps, samples=a.samples, seed=a.seed,
                                  init=str(init[0]), ic=str(a.ic), fcn3_precision=a.fcn3_precision,
                                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                  torchdynamo_disabled=os.environ.get('TORCHDYNAMO_DISABLE', '0') == '1')
    (a.output / 'timings.json').write_text(json.dumps(timings, indent=2))
    print(json.dumps(timings['summary'], indent=2))


if __name__ == '__main__':
    main()
