"""Verify the FCN3 Kong-rey centre track against IBTrACS best tracks (JMA = WMO agency, and JTWC).

The FCN3 centre is the grid-point MSLP minimum of the saved Taiwan-region subset (0.25 degree), so
position errors carry up to half a grid diagonal (~0.18 degree, ~19 km here) of grid quantization. JMA pressures pair with 10-minute winds and
JTWC pressures with 1-minute winds; the two agencies' intensities are not interchangeable.
"""
import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
COLORS = {'FCN3': '#2a78d6', 'JMA': INK, 'JTWC': '#eb6834'}
EARTH_RADIUS_KM = 6371.0


def distance_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


def fcn3_track(path):
    f = xr.open_dataarray(path)
    init = pd.Timestamp(f.attrs['init'][:19])
    rows = []
    for lead in f.lead_time.values:
        msl = f.sel(variable='msl', lead_time=lead)
        i = np.unravel_index(int(msl.argmin()), msl.shape)
        rows.append(dict(lead_h=int(lead), time=init + pd.Timedelta(hours=int(lead)), fcn3_lat=float(msl.lat[i[0]]),
                         fcn3_lon=float(msl.lon[i[1]]), fcn3_pres=float(msl.min()) / 100))
    return init, pd.DataFrame(rows)


def best_track(path, sid):
    bt = pd.read_csv(path, skiprows=[1], keep_default_na=False, low_memory=False)
    bt = bt[bt.SID == sid].copy()
    bt['time'] = pd.to_datetime(bt.ISO_TIME)
    for c in ['TOKYO_LAT', 'TOKYO_LON', 'TOKYO_PRES', 'USA_LAT', 'USA_LON', 'USA_PRES', 'DIST2LAND']:
        bt[c] = pd.to_numeric(bt[c], errors='coerce')
    return bt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bt', type=Path, required=True, help='IBTrACS CSV (full file or Kong-rey extract)')
    p.add_argument('--sid', default='2024298N13150')
    p.add_argument('--fcn3', type=Path, required=True, help='fcn3_taiwan_region.nc')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    init, fc = fcn3_track(a.fcn3)
    bt = best_track(a.bt, a.sid)
    df = fc.merge(bt[['time', 'TOKYO_LAT', 'TOKYO_LON', 'TOKYO_PRES', 'USA_LAT', 'USA_LON', 'USA_PRES', 'DIST2LAND']],
                  on='time', how='left')
    if df[['TOKYO_LAT', 'USA_LAT']].isna().any().any():
        raise SystemExit('best track missing at a forecast valid time')
    df['err_jma_km'] = distance_km(df.fcn3_lat, df.fcn3_lon, df.TOKYO_LAT, df.TOKYO_LON)
    df['err_jtwc_km'] = distance_km(df.fcn3_lat, df.fcn3_lon, df.USA_LAT, df.USA_LON)
    df['pres_err_jma_hpa'] = df.fcn3_pres - df.TOKYO_PRES
    df['pres_err_jtwc_hpa'] = df.fcn3_pres - df.USA_PRES
    df.to_csv(a.output / 'kongrey_track_verification.csv', index=False, float_format='%.3f')

    window = bt[(bt.time >= init - pd.Timedelta(hours=6)) & (bt.time <= df.time.max() + pd.Timedelta(hours=6))]
    land = window[(window.DIST2LAND == 0) & window.TOKYO_LAT.between(21.5, 25.5)]
    landfall = land.iloc[0] if len(land) else None
    lines = [f'FCN3 init {init} UTC vs IBTrACS {a.sid} (JMA = WMO agency; JTWC = USA_*)',
             f'{"lead":>4s} {"valid UTC":16s} {"FCN3":>22s} {"JMA":>18s} {"JTWC":>18s} {"err JMA":>8s} {"err JTWC":>9s}'
             f' {"dP JMA":>7s} {"dP JTWC":>8s}']
    for r in df.itertuples():
        lines.append(f'{r.lead_h:4d} {r.time:%Y-%m-%d %H} {r.fcn3_pres:6.1f}@{r.fcn3_lat:5.2f},{r.fcn3_lon:6.2f}'
                     f' {r.TOKYO_PRES:4.0f}@{r.TOKYO_LAT:4.1f},{r.TOKYO_LON:5.1f} {r.USA_PRES:4.0f}@{r.USA_LAT:4.1f},'
                     f'{r.USA_LON:5.1f} {r.err_jma_km:7.0f}km {r.err_jtwc_km:7.0f}km {r.pres_err_jma_hpa:+7.1f}'
                     f' {r.pres_err_jtwc_hpa:+8.1f}')
    fcst = df[df.lead_h > 0]
    lines.append(f'mean track error, leads +6..+{df.lead_h.max()} h: {fcst.err_jma_km.mean():.0f} km vs JMA, '
                 f'{fcst.err_jtwc_km.mean():.0f} km vs JTWC')
    lines.append(f'initial-condition centre error (GFS analysis, +0 h): {df.err_jma_km.iloc[0]:.0f} km vs JMA; '
                 f'central pressure {df.fcn3_pres.iloc[0]:.1f} vs JMA {df.TOKYO_PRES.iloc[0]:.0f} / '
                 f'JTWC {df.USA_PRES.iloc[0]:.0f} hPa')
    if landfall is not None:
        lines.append(f'first best-track point over Taiwan land (DIST2LAND=0, 3-hourly): {landfall.time:%Y-%m-%d %H} UTC '
                     f'at {landfall.TOKYO_LAT:.1f}N {landfall.TOKYO_LON:.1f}E (JMA)')
    text = '\n'.join(lines)
    print(text)
    (a.output / 'kongrey_track_verification.txt').write_text(text + '\n')
    plot(df, window, landfall, init, a.output / 'kongrey_track_verification.png')


def plot(df, window, landfall, init, out):
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK_2, 'axes.labelcolor': INK_2, 'xtick.color': INK_2,
                         'ytick.color': INK_2, 'axes.titlecolor': INK, 'axes.titlesize': 10})
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ax.set_extent([117, 127, 18, 28], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='#f3f2ef', edgecolor=INK_2,
                                                linewidth=0.6))
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color=GRID)
    gl.top_labels = gl.right_labels = False
    tr = ccrs.PlateCarree()
    ax.plot(window.TOKYO_LON, window.TOKYO_LAT, '-', color=COLORS['JMA'], lw=2, transform=tr, label='JMA best track')
    ax.plot(window.USA_LON, window.USA_LAT, '--', color=COLORS['JTWC'], lw=2, transform=tr, label='JTWC best track')
    ax.plot(df.fcn3_lon, df.fcn3_lat, '-', color=COLORS['FCN3'], lw=2.5, transform=tr, label='FCN3 (grid-point min MSLP)')
    for name, lon, lat in (('JMA', df.TOKYO_LON, df.TOKYO_LAT), ('FCN3', df.fcn3_lon, df.fcn3_lat)):
        ax.scatter(lon, lat, s=26, color=COLORS[name], edgecolor='white', lw=1.2, zorder=4, transform=tr)
    for r in df.iloc[::2].itertuples():
        ax.plot([r.fcn3_lon, r.TOKYO_LON], [r.fcn3_lat, r.TOKYO_LAT], ':', color=INK_2, lw=0.8, transform=tr)
        ax.annotate(f'+{r.lead_h}h', (r.fcn3_lon, r.fcn3_lat), xytext=(7, 0), textcoords='offset points', fontsize=7,
                    color=COLORS['FCN3'], va='center', transform=tr)
    if landfall is not None:
        ax.scatter([landfall.TOKYO_LON], [landfall.TOKYO_LAT], marker='*', s=150, color=INK, edgecolor='white', zorder=5,
                   transform=tr)
        ax.annotate(f'over land by {landfall.time:%m-%d %H}Z', (landfall.TOKYO_LON, landfall.TOKYO_LAT),
                    xytext=(-12, -14), textcoords='offset points', ha='right', fontsize=7, color=INK, transform=tr)
    ax.set_title(f'(a) Typhoon Kong-rey track, FCN3 init {init:%Y-%m-%d %H} UTC', loc='left')
    ax.legend(frameon=False, loc='upper right', fontsize=8)

    leads = df.lead_h
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(leads, df.err_jma_km, 'o-', color=COLORS['JMA'], lw=2, ms=6, label='vs JMA')
    ax.plot(leads, df.err_jtwc_km, 's--', color=COLORS['JTWC'], lw=2, ms=6, label='vs JTWC')
    quant_km = float(distance_km(23.0, 121.0, 23.125, 121.125))   # half a 0.25-degree cell diagonal
    ax.axhline(quant_km, color=INK_2, lw=0.8, ls=':')
    ax.annotate(f'grid quantization up to ≈ {quant_km:.0f} km', (leads.iloc[0], quant_km), xytext=(0, -10),
                textcoords='offset points', ha='left', fontsize=7, color=INK_2)
    ax.set_ylim(bottom=0); ax.set_xticks(leads)
    ax.set_xlabel('forecast lead (h)'); ax.set_ylabel('km')
    ax.set_title('(b) FCN3 centre position error', loc='left')
    ax.legend(frameon=False, fontsize=8, loc='upper left')

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(leads, df.TOKYO_PRES, 'o-', color=COLORS['JMA'], lw=2, ms=6, label='JMA (10-min wind basis)')
    ax.plot(leads, df.USA_PRES, 's--', color=COLORS['JTWC'], lw=2, ms=6, label='JTWC (1-min wind basis)')
    ax.plot(leads, df.fcn3_pres, 'o-', color=COLORS['FCN3'], lw=2, ms=6, label='FCN3 0.25° grid minimum')
    ax.set_xticks(leads); ax.set_xlabel('forecast lead (h)'); ax.set_ylabel('hPa')
    ax.set_title('(c) Central pressure', loc='left')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    for axis in fig.axes[1:]:
        axis.grid(axis='y', color=GRID, lw=0.6); axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    fig.text(0.01, 0.01, 'Best tracks: IBTrACS v04r01 (NOAA NCEI), SID 2024298N13150. FCN3: Earth2Studio 0.20, BF16, '
             'GFS analysis initial condition, seed 20241030.', fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
