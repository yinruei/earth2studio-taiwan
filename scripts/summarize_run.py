"""Print the FCN3 typhoon centre (min MSLP) per lead and CorrDiff Taiwan sample statistics."""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

run = Path(sys.argv[1])
f = xr.open_dataarray(run / 'fcn3_taiwan_region.nc')
print('FCN3 region', dict(f.sizes))
for lead in f.lead_time.values:
    msl = f.sel(variable='msl', lead_time=lead)
    i = np.unravel_index(int(msl.argmin()), msl.shape)
    wind = np.hypot(f.sel(variable='u10m', lead_time=lead), f.sel(variable='v10m', lead_time=lead))
    print(f'FCN3 +{int(lead):2d}h  min MSLP {float(msl.min()) / 100:7.1f} hPa at '
          f'{float(msl.lat[i[0]]):5.2f}N {float(msl.lon[i[1]]):6.2f}E  max 10 m wind {float(wind.max()):5.1f} m/s')

c = xr.open_dataarray(run / 'corrdiff_taiwan.nc')
print('CorrDiff', dict(c.sizes), list(c['variable'].values),
      f'lat {float(c.lat.min()):.2f}-{float(c.lat.max()):.2f} lon {float(c.lon.min()):.2f}-{float(c.lon.max()):.2f}')
for lead in c.lead_time.values:
    x = c.sel(lead_time=lead)
    wind = np.hypot(x.sel(variable='u10m'), x.sel(variable='v10m')).max(dim=('y', 'x')).values
    mrr = x.sel(variable='mrr').max(dim=('y', 'x')).values
    t2m = x.sel(variable='t2m')
    print(f'CorrDiff +{int(lead):2d}h  max 10 m wind per sample {np.round(wind, 1)}  max mrr {np.round(mrr, 1)}  '
          f't2m {float(t2m.min()):.1f}-{float(t2m.max()):.1f} K  all finite {bool(np.isfinite(x).all())}')
