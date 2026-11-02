"""Figures for the Kong-rey FCN3 -> CorrDiff Taiwan run: downscaled fields and compile/eager timings."""
import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SERIES = {'compiled': '#2a78d6', 'eager': '#eb6834'}   # categorical slots 1, 2
EXTENT = [116.0, 126.0, 18.8, 28.0]
LAND = cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='none', edgecolor=INK, linewidth=0.6)
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK_2, 'axes.labelcolor': INK_2, 'xtick.color': INK_2,
                     'ytick.color': INK_2, 'axes.titlecolor': INK, 'axes.titlesize': 10})


def track(fcn3):
    rows = []
    for lead in fcn3.lead_time.values:
        msl = fcn3.sel(variable='msl', lead_time=lead)
        i = np.unravel_index(int(msl.argmin()), msl.shape)
        rows.append((int(lead), float(msl.lon[i[1]]), float(msl.lat[i[0]]), float(msl.min()) / 100))
    return rows


def map_axes(fig, pos, title):
    ax = fig.add_subplot(pos, projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(LAND)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color=GRID)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 7, 'color': INK_2}
    ax.set_title(title, loc='left')
    return ax


def fields_figure(run, lead, out):
    fcn3 = xr.open_dataarray(run / 'fcn3_taiwan_region.nc')
    cd = xr.open_dataarray(run / 'corrdiff_taiwan.nc')
    init = np.datetime64(fcn3.attrs['init'][:19])
    valid = str(init + np.timedelta64(lead, 'h'))[:13].replace('T', ' ') + ' UTC'
    f = fcn3.sel(lead_time=lead)
    c = cd.sel(lead_time=lead)
    f_wind = np.hypot(f.sel(variable='u10m'), f.sel(variable='v10m'))
    c_wind = np.hypot(c.sel(variable='u10m'), c.sel(variable='v10m'))
    vmax = float(np.ceil(max(f_wind.max(), c_wind.max()) / 5) * 5)
    pts = track(fcn3)

    fig = plt.figure(figsize=(11, 9.6))
    ax = map_axes(fig, 221, f'(a) FCN3 0.25°: 10 m wind and MSLP, +{lead} h')
    m = ax.pcolormesh(f.lon, f.lat, f_wind, cmap='Blues', vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), shading='auto')
    cs = ax.contour(f.lon, f.lat, f.sel(variable='msl') / 100, levels=np.arange(940, 1024, 4), colors=INK_2,
                    linewidths=0.5, transform=ccrs.PlateCarree())
    ax.clabel(cs, fontsize=6, fmt='%d')
    ax.plot([p[1] for p in pts], [p[2] for p in pts], '-', color=INK, lw=1.2, transform=ccrs.PlateCarree())
    ax.scatter([p[1] for p in pts], [p[2] for p in pts], s=22, color='white', edgecolor=INK, lw=1, zorder=3,
               transform=ccrs.PlateCarree())
    for h, lon, lat, p in pts[::2]:
        ax.annotate(f'+{h}h {p:.0f}', (lon, lat), xytext=(6, -3), textcoords='offset points', fontsize=7, color=INK,
                    transform=ccrs.PlateCarree())
    fig.colorbar(m, ax=ax, shrink=0.8, label='10 m wind speed (m/s)')

    ax = map_axes(fig, 222, f'(b) CorrDiff Taiwan ~2 km: 10 m wind, sample 0')
    m = ax.pcolormesh(c.lon, c.lat, c_wind.isel(sample=0), cmap='Blues', vmin=0, vmax=vmax,
                      transform=ccrs.PlateCarree(), shading='auto')
    fig.colorbar(m, ax=ax, shrink=0.8, label='10 m wind speed (m/s)')

    ax = map_axes(fig, 223, '(c) CorrDiff: maximum radar reflectivity, sample 0')
    mrr = c.sel(variable='mrr').isel(sample=0).where(lambda x: x >= 5)
    m = ax.pcolormesh(c.lon, c.lat, mrr, cmap='Purples', vmin=5, vmax=60, transform=ccrs.PlateCarree(), shading='auto')
    fig.colorbar(m, ax=ax, shrink=0.8, label='mrr (dBZ, < 5 blank)')

    ax = map_axes(fig, 224, f'(d) CorrDiff: 10 m wind spread across {cd.sizes["sample"]} samples')
    m = ax.pcolormesh(c.lon, c.lat, c_wind.std(dim='sample'), cmap='Oranges', vmin=0,
                      transform=ccrs.PlateCarree(), shading='auto')
    fig.colorbar(m, ax=ax, shrink=0.8, label='sample std of 10 m wind (m/s)')

    fig.suptitle(f'Typhoon Kong-rey: GFS {str(init)[:13].replace("T", " ")} UTC → FCN3 → CorrDiff Taiwan, '
                 f'valid {valid}', x=0.02, ha='left', fontsize=12, color=INK)
    fig.text(0.02, 0.01, 'Earth2Studio 0.20, offline on 1x A100. GFS analysis initial condition; no observations. '
             'Centre = FCN3 grid-point minimum MSLP (0.25°); not a best track.', fontsize=7, color=INK_2)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.05, wspace=0.18, hspace=0.16)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def timing_figure(compiled, eager, out):
    t = {k: json.loads((p / 'timings.json').read_text()) for k, p in (('compiled', compiled), ('eager', eager))}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
    leads = [s['lead_hours'] for s in t['eager']['steps']]
    for name, tt in t.items():
        ax1.plot(leads, [s['fcn3_sec'] for s in tt['steps']], 'o-', color=SERIES[name], lw=2, ms=6, label=name)
    ax1.set_yscale('log')
    ax1.set_ylim(1.5, 100)
    ax1.set_yticks([2, 5, 10, 20, 50])
    ax1.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xticks(leads)
    ax1.set_xlabel('forecast lead (h)')
    ax1.set_ylabel('FCN3 step wall time (s, log)')
    ax1.set_title('(a) FCN3 time per 6 h step', loc='left')
    for name in ('compiled', 'eager'):
        s = t[name]['steps'][1]['fcn3_sec']
        ax1.annotate(f'{s:.1f} s', (6, s), xytext=(8, 0), textcoords='offset points', color=INK_2, fontsize=8,
                     va='center')
    ax1.annotate(f'steady {t["eager"]["summary"]["fcn3_step_median_sec"]:.2f} s (both)', (30, 1.96),
                 xytext=(0, 10), textcoords='offset points', color=INK_2, fontsize=8, ha='center')

    groups = [('FCN3 step\n(median, steady)', 'summary', 'fcn3_step_median_sec'),
              ('CorrDiff 4 samples\n(median per lead)', 'summary', 'corrdiff_per_lead_median_sec')]
    _bars(ax2, t, groups, '(b) Steady-state cost', 'seconds')
    one_off = [('load FCN3', None, 'fcn3_load_sec'), ('load CorrDiff', None, 'corrdiff_load_sec'),
               ('first FCN3\nstep', 'first', None)]
    _bars(ax3, t, one_off, '(c) One-time cost', 'seconds')
    for ax in (ax1, ax2, ax3):
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
    fig.suptitle('torch.compile vs eager, same node / seed / initial condition (1 run each)', x=0.01, ha='left',
                 fontsize=11, color=INK)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2, frameon=False, bbox_to_anchor=(0.99, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _bars(ax, t, groups, title, ylabel):
    x = np.arange(len(groups))
    for k, name in enumerate(('compiled', 'eager')):
        vals = []
        for _, section, key in groups:
            if section == 'first':
                vals.append(t[name]['steps'][1]['fcn3_sec'])
            elif section:
                vals.append(t[name][section][key])
            else:
                vals.append(t[name][key])
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=SERIES[name], label=name, edgecolor='white', lw=2)
        ax.bar_label(bars, fmt='%.2f' if max(vals) < 5 else '%.1f', fontsize=7, color=INK_2, padding=2)
    ax.set_xticks(x, [g[0] for g in groups], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left')
    ax.set_ylim(0, max(ax.get_ylim()[1], 0) * 1.12)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiled', type=Path, required=True)
    p.add_argument('--eager', type=Path, required=True)
    p.add_argument('--lead', type=int, default=30)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    fields_figure(a.eager, a.lead, a.output / f'kongrey_fields_lead{a.lead:02d}h.png')
    timing_figure(a.compiled, a.eager, a.output / 'compile_vs_eager_timing.png')
    print('wrote', sorted(x.name for x in a.output.iterdir()))


if __name__ == '__main__':
    main()
