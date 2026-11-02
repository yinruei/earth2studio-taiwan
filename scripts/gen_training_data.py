"""Generate a small, publishable training set from FCN3: regional 64x64 trajectories near Taiwan.

Each member is an FCN3 rollout from the same GFS analysis with a different seed (FCN3 is stochastic), so the
set has little meteorological diversity. It exists to measure distributed-training performance, not to train
a skilful model. One worker per GPU; the job script splits members round-robin.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

VARIABLES = ['u10m', 'v10m', 't2m', 'msl', 'tcwv', 'z500', 't850', 'q850']
LAT = np.arange(31.75, 15.99, -0.25)   # 64 points, north to south like FCN3
LON = np.arange(112.0, 128.0, 0.25)    # 64 points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ic', type=Path, required=True)
    p.add_argument('--fcn3', type=Path, required=True)
    p.add_argument('--members', type=int, nargs='+', required=True)
    p.add_argument('--nsteps', type=int, default=16)
    p.add_argument('--base-seed', type=int, default=20241030)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from earth2studio.data import DataArrayFile, fetch_data
    from earth2studio.models.auto import Package
    from earth2studio.models.px import FCN3
    from earth2studio.utils.coords import map_coords
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    fcn3 = FCN3.load_model(Package(str(a.fcn3))).to(device)
    data = DataArrayFile(str(a.ic))
    ic = fcn3.input_coords()
    x0, coords0 = fetch_data(data, time=data.da['time'].values[:1], variable=ic['variable'],
                             lead_time=ic['lead_time'], device=device)
    x0, coords0 = map_coords(x0, coords0, ic)
    var_idx = [list(ic['variable']).index(v) for v in VARIABLES]
    lat_idx = [int(np.argmin(np.abs(ic['lat'] - v))) for v in LAT]
    lon_idx = [int(np.argmin(np.abs(ic['lon'] - v))) for v in LON]
    if not (np.allclose(ic['lat'][lat_idx], LAT) and np.allclose(ic['lon'][lon_idx], LON)):
        raise SystemExit('region does not fall on the FCN3 grid')
    sel = (torch.tensor(var_idx, device=device), torch.tensor(lat_idx, device=device),
           torch.tensor(lon_idx, device=device))
    for m in a.members:
        t0 = time.time()
        fcn3.set_rng(seed=a.base_seed + m)
        model = fcn3.create_iterator(x0.clone(), coords0)
        frames = []
        for _ in range(a.nsteps + 1):
            x, _ = next(model)
            field = x.reshape(-1, *x.shape[-3:])[0]                    # (variable, lat, lon)
            field = field.index_select(0, sel[0]).index_select(1, sel[1]).index_select(2, sel[2])
            frames.append(field.float().cpu().numpy())
        arr = np.stack(frames)                                       # (nsteps + 1, 8, 64, 64)
        if not np.isfinite(arr).all():
            raise SystemExit(f'member {m}: non-finite values')
        np.save(a.output / f'member_{m:02d}.npy', arr)
        print(json.dumps(dict(member=m, seconds=round(time.time() - t0, 1), shape=list(arr.shape))), flush=True)
    meta = dict(variables=VARIABLES, lat=LAT.tolist(), lon=LON.tolist(), step_hours=6, nsteps=a.nsteps,
                base_seed=a.base_seed, source='FCN3 (Earth2Studio 0.20) from GFS analysis 2024-10-30 00 UTC')
    (a.output / 'meta.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
