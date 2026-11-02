"""FCN3 BF16 autocast vs FP32: step timings, regional t2m/MSLP deviation and typhoon-centre drift.

Runs are grouped by the fcn3_precision recorded in timings.json. Repeated runs of the same
precision are checked for bit-identical output (same seed) before any comparison.
"""
import argparse
import csv
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
SERIES = {'fp32': '#2a78d6', 'bf16': '#eb6834'}   # categorical slots 1, 2
EARTH_RADIUS_KM = 6371.0


def centre(fcn3, lead):
    msl = fcn3.sel(variable='msl', lead_time=lead)
    i = np.unravel_index(int(msl.argmin()), msl.shape)
    return float(msl.lat[i[0]]), float(msl.lon[i[1]]), float(msl.min()) / 100


def distance_km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


def weighted_rmse(diff):
    w = np.cos(np.radians(diff.lat))
    return float(np.sqrt((diff ** 2).weighted(w).mean(dim=('lat', 'lon'))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    runs = {'fp32': [], 'bf16': []}
    for run in a.runs:
        t = json.loads((run / 'timings.json').read_text())
        runs[t['environment']['fcn3_precision']].append((run, t))
    if not runs['fp32'] or not runs['bf16']:
        raise SystemExit('need at least one fp32 and one bf16 run')

    lines = []
    def say(text=''):
        print(text)
        lines.append(text)

    say('## Timing (median of steady FCN3 steps = leads +12..+48 h; CorrDiff median per lead)')
    say(f'{"run":36s}{"precision":>10s}{"FCN3 step s":>13s}{"first step s":>14s}{"CorrDiff s":>12s}{"peak GiB":>10s}')
    steady = {}
    for prec, items in runs.items():
        for run, t in items:
            fsteady = float(np.median([s['fcn3_sec'] for s in t['steps'][2:]]))
            steady.setdefault(prec, []).append(fsteady)
            say(f'{run.name:36s}{prec:>10s}{fsteady:13.3f}{t["steps"][1]["fcn3_sec"]:14.2f}'
                f'{t["summary"]["corrdiff_per_lead_median_sec"]:12.3f}{t["peak_allocated_gib_during_rollout"]:10.2f}')
    speedup = np.mean(steady['fp32']) / np.mean(steady['bf16'])
    say(f'FCN3 steady-step speedup, BF16 vs FP32 (mean of runs): {speedup:.2f}x '
        f'({np.mean(steady["fp32"]):.3f} s -> {np.mean(steady["bf16"]):.3f} s)')

    fields = {prec: [xr.open_dataarray(r / 'fcn3_taiwan_region.nc') for r, _ in items] for prec, items in runs.items()}
    say('\n## Reproducibility (same precision, same seed)')
    for prec, fs in fields.items():
        for k in range(1, len(fs)):
            say(f'{prec}: run 1 vs run {k + 1} max |diff| = {float(np.abs(fs[0] - fs[k]).max()):.3g}')

    f32, b16 = fields['fp32'][0], fields['bf16'][0]
    say('\n## BF16 - FP32, FCN3 Taiwan region (15-32N, 112-130E), cos-lat weighted RMSE and max |diff|')
    say(f'{"lead":>5s}{"t2m RMSE K":>12s}{"t2m max K":>11s}{"msl RMSE hPa":>14s}{"msl max hPa":>13s}'
        f'{"centre FP32":>22s}{"centre BF16":>22s}{"dist km":>9s}{"dP hPa":>8s}')
    rows = []
    for lead in f32.lead_time.values:
        dt = b16.sel(variable='t2m', lead_time=lead) - f32.sel(variable='t2m', lead_time=lead)
        dp = (b16.sel(variable='msl', lead_time=lead) - f32.sel(variable='msl', lead_time=lead)) / 100
        c32, c16 = centre(f32, lead), centre(b16, lead)
        row = dict(lead_h=int(lead), t2m_rmse_k=weighted_rmse(dt), t2m_maxabs_k=float(np.abs(dt).max()),
                   msl_rmse_hpa=weighted_rmse(dp), msl_maxabs_hpa=float(np.abs(dp).max()),
                   fp32_lat=c32[0], fp32_lon=c32[1], fp32_pmin_hpa=c32[2],
                   bf16_lat=c16[0], bf16_lon=c16[1], bf16_pmin_hpa=c16[2],
                   centre_distance_km=float(distance_km(c32, c16)), pmin_diff_hpa=c16[2] - c32[2])
        rows.append(row)
        say(f'{row["lead_h"]:5d}{row["t2m_rmse_k"]:12.4f}{row["t2m_maxabs_k"]:11.3f}{row["msl_rmse_hpa"]:14.4f}'
            f'{row["msl_maxabs_hpa"]:13.3f}{f"{c32[2]:.1f}@{c32[0]:.2f}N,{c32[1]:.2f}E":>22s}'
            f'{f"{c16[2]:.1f}@{c16[0]:.2f}N,{c16[1]:.2f}E":>22s}{row["centre_distance_km"]:9.1f}{row["pmin_diff_hpa"]:8.2f}')
    with (a.output / 'bf16_vs_fp32.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    cd32, cd16 = (xr.open_dataarray(runs[p][0][0] / 'corrdiff_taiwan.nc') for p in ('fp32', 'bf16'))
    say('\n## Downstream CorrDiff (its own precision unchanged; inputs differ), sample-mean fields')
    for lead in cd32.lead_time.values:
        m32, m16 = cd32.sel(lead_time=lead).mean('sample'), cd16.sel(lead_time=lead).mean('sample')
        dt = m16.sel(variable='t2m') - m32.sel(variable='t2m')
        w32 = np.hypot(m32.sel(variable='u10m'), m32.sel(variable='v10m'))
        w16 = np.hypot(m16.sel(variable='u10m'), m16.sel(variable='v10m'))
        say(f'  +{int(lead):2d}h  t2m RMSE {float(np.sqrt((dt ** 2).mean())):.3f} K   10 m wind RMSE '
            f'{float(np.sqrt(((w16 - w32) ** 2).mean())):.3f} m/s   max wind {float(w32.max()):.1f} vs {float(w16.max()):.1f}')
    (a.output / 'bf16_vs_fp32.txt').write_text('\n'.join(lines) + '\n')
    plot(runs, rows, f32, a.output / 'bf16_vs_fp32.png')


def plot(runs, rows, f32, out):
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK_2, 'axes.labelcolor': INK_2, 'xtick.color': INK_2,
                         'ytick.color': INK_2, 'axes.titlecolor': INK, 'axes.titlesize': 10})
    fig = plt.figure(figsize=(14, 8.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15])
    leads = [r['lead_h'] for r in rows]

    ax = fig.add_subplot(gs[0, 0])
    for prec, items in runs.items():
        for k, (_, t) in enumerate(items):
            steps = t['steps'][2:]
            ax.plot([s['lead_hours'] for s in steps], [s['fcn3_sec'] for s in steps], 'o-' if k == 0 else 's--',
                    color=SERIES[prec], lw=2, ms=6, label=f'{prec.upper()} run {k + 1}')
    ax.set_ylim(bottom=0)
    ax.set_xticks(leads[2:])
    ax.set_xlabel('forecast lead (h)'); ax.set_ylabel('seconds')
    ax.set_title('(a) FCN3 steady step time', loc='left')
    ax.legend(frameon=False, fontsize=8, loc='lower right')

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(leads, [r['t2m_rmse_k'] for r in rows], 'o-', color=INK, lw=2, ms=6, label='RMSE')
    ax.plot(leads, [r['t2m_maxabs_k'] for r in rows], 'o--', color=INK_2, lw=1.5, ms=5, label='max |diff|')
    ax.set_xticks(leads); ax.set_xlabel('forecast lead (h)'); ax.set_ylabel('K')
    ax.set_title('(b) t2m: BF16 - FP32 (region)', loc='left')
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(leads, [r['msl_rmse_hpa'] for r in rows], 'o-', color=INK, lw=2, ms=6, label='RMSE')
    ax.plot(leads, [r['msl_maxabs_hpa'] for r in rows], 'o--', color=INK_2, lw=1.5, ms=5, label='max |diff|')
    ax.set_xticks(leads); ax.set_xlabel('forecast lead (h)'); ax.set_ylabel('hPa')
    ax.set_title('(d) MSLP: BF16 - FP32 (region)', loc='left')
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    diffs = [r['pmin_diff_hpa'] for r in rows]
    ax.bar(leads, diffs, width=3.5, color=INK_2)
    ax.axhline(0, color=INK_2, lw=0.8)
    for lead, d in zip(leads, diffs):
        ax.annotate(f'{d:+.2f}', (lead, d), xytext=(0, 3 if d >= 0 else -10), textcoords='offset points',
                    ha='center', fontsize=7, color=INK_2)
    max_km = max(r['centre_distance_km'] for r in rows)
    ax.set_xticks(leads); ax.set_xlabel('forecast lead (h)'); ax.set_ylabel('hPa')
    ax.set_title(f'(c) Centre min MSLP: BF16 - FP32\n(centre separation {max_km:.0f} km at every lead, 0.25° grid)',
                 loc='left')
    lim = max(0.5, max(abs(d) for d in diffs) * 1.4)
    ax.set_ylim(-lim, lim)

    ax = fig.add_subplot(gs[:, 2], projection=ccrs.PlateCarree())
    ax.set_extent([117, 127, 18.5, 27.5], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='none', edgecolor=INK, linewidth=0.6))
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color=GRID)
    gl.top_labels = gl.right_labels = False
    # FP32 drawn wide underneath, BF16 thin and dashed on top, so identical tracks stay visible
    for prec, lw, ls, size in (('fp32', 6, '-', 70), ('bf16', 1.8, '--', 18)):
        lat = [r[f'{prec}_lat'] for r in rows]; lon = [r[f'{prec}_lon'] for r in rows]
        ax.plot(lon, lat, ls, color=SERIES[prec], lw=lw, transform=ccrs.PlateCarree(), label=prec.upper(),
                zorder=2 if prec == 'fp32' else 4)
        ax.scatter(lon, lat, s=size, color=SERIES[prec], edgecolor='white', lw=1.2,
                   zorder=3 if prec == 'fp32' else 5, transform=ccrs.PlateCarree())
    for r in rows[::2]:
        ax.annotate(f'+{r["lead_h"]}h', (r['fp32_lon'], r['fp32_lat']), xytext=(7, -3), textcoords='offset points',
                    fontsize=7, color=INK, transform=ccrs.PlateCarree())
    ax.set_title('(e) FCN3 centre track (grid-point min MSLP, 0.25°)', loc='left')
    ax.legend(frameon=False, loc='upper right')
    fig.suptitle('FCN3 BF16 autocast vs FP32: Typhoon Kong-rey, init 2024-10-30 00 UTC, same seed, 1x A100',
                 x=0.01, ha='left', fontsize=12, color=INK)
    for axis in fig.axes[:4]:
        axis.grid(axis='y', color=GRID, lw=0.6); axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
