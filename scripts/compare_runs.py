"""Compare two FCN3 -> CorrDiff runs (e.g. torch.compile vs eager): stage timings and output drift."""
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

a, b = Path(sys.argv[1]), Path(sys.argv[2])
ta, tb = (json.loads((p / 'timings.json').read_text()) for p in (a, b))
print(f'{"":28s}{a.name:>28s}{b.name:>28s}')
for key in ['fcn3_load_sec', 'corrdiff_load_sec', 'peak_allocated_gib_during_rollout']:
    print(f'{key:28s}{ta[key]:28.3f}{tb[key]:28.3f}')
for key in ['fcn3_step_median_sec', 'corrdiff_per_lead_median_sec', 'total_rollout_sec']:
    print(f'{key:28s}{ta["summary"][key]:28.3f}{tb["summary"][key]:28.3f}')
print(f'FCN3 step speedup ({b.name} / {a.name}): {tb["summary"]["fcn3_step_median_sec"] / ta["summary"]["fcn3_step_median_sec"]:.2f}x')
print('per-step fcn3_sec / corrdiff_sec:')
for sa, sb in zip(ta['steps'], tb['steps']):
    print(f'  +{sa["lead_hours"]:2d}h  {sa["fcn3_sec"]:7.3f} / {sa["corrdiff_sec"]:6.3f}    {sb["fcn3_sec"]:7.3f} / {sb["corrdiff_sec"]:6.3f}')

fa, fb = (xr.open_dataarray(p / 'fcn3_taiwan_region.nc') for p in (a, b))
print('FCN3 region max |difference| per lead (same seed, same initial condition):')
for lead in fa.lead_time.values:
    diffs = {v: float(np.abs(fa.sel(variable=v, lead_time=lead) - fb.sel(variable=v, lead_time=lead)).max())
             for v in fa['variable'].values}
    centre = []
    for f in (fa, fb):
        msl = f.sel(variable='msl', lead_time=lead)
        i = np.unravel_index(int(msl.argmin()), msl.shape)
        centre.append(f'{float(msl.min()) / 100:.1f}hPa@{float(msl.lat[i[0]]):.2f}N,{float(msl.lon[i[1]]):.2f}E')
    print(f'  +{int(lead):2d}h  ' + '  '.join(f'{k} {v:.3g}' for k, v in diffs.items()) + f'   centre {centre[0]} vs {centre[1]}')
