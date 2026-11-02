"""Convert the byte-range GFS f000 GRIB subset into an Earth2Studio DataArrayFile.

Each message is matched against manifest.json (variable order and GRIB name/level)
and converted with Earth2Studio's own GFSLexicon modifier, so units match what the
online GFS data source would return (e.g. HGT gpm -> geopotential m2/s2).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pygrib
import xarray as xr
from earth2studio.lexicon import GFSLexicon


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grib', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    messages = pygrib.open(str(a.grib))
    fields, lat, lon = [], None, None
    for entry, msg in zip(manifest['messages'], messages, strict=True):
        variable = entry['variable']
        key, modifier = GFSLexicon.get_item(variable)
        name, level = entry['idx_line'].split(':')[3:5]
        if key != f'{name}::{level}' or key != entry['lexicon']:
            raise ValueError(f'{variable}: lexicon {key} does not match message {name}::{level}')
        values = msg.values.astype(np.float64)
        if lat is None:
            lats, lons = msg.latlons()
            lat, lon = lats[:, 0], lons[0, :]
        if values.shape != (721, 1440) or not np.isfinite(values).all():
            raise ValueError(f'{variable}: unexpected shape {values.shape} or non-finite values')
        fields.append(modifier(values).astype(np.float32))
    if not (np.allclose(lat, np.linspace(90, -90, 721)) and np.allclose(lon, np.arange(1440) * 0.25)):
        raise ValueError('GFS grid is not the expected 0.25-degree 90..-90 / 0..359.75 grid')
    time = np.array([np.datetime64(manifest['init'].replace('Z', ''), 'ns')])
    da = xr.DataArray(np.stack(fields)[None], dims=['time', 'variable', 'lat', 'lon'],
                      coords=dict(time=time, variable=manifest['variables'],
                                  lat=np.linspace(90, -90, 721), lon=np.arange(1440) * 0.25),
                      name='fields', attrs=dict(source=manifest['source'], converter='make_ic.py + GFSLexicon modifiers'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(a.output)
    for v in ['t2m', 'msl', 'z500', 'tcwv', 'q850']:
        x = da.sel(variable=v).values
        print(f'{v:5s} min {x.min():12.4f} mean {x.mean():12.4f} max {x.max():12.4f}')
    print(f'wrote {a.output} {da.shape}')


if __name__ == '__main__':
    main()
