"""Summarize and plot ensemble_scaling_job.sh: speedup, start-up cost, per-member time, GPU timeline, identity.

End-to-end time runs from launching the workers to the last member finishing (includes Python start-up,
model loading and, for the shm variant, staging the weights). Compute time runs from the first worker
starting its first member to the last member finishing.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
ORDER = ['n1', 'n2', 'n4', 'n4_numa', 'n4_shm']
LABELS = {'n1': '1 GPU', 'n2': '2 GPUs', 'n4': '4 GPUs', 'n4_numa': '4 GPUs, NUMA-bound', 'n4_shm': '4 GPUs, weights in /dev/shm'}
SHORT = {'n1': '1 GPU', 'n2': '2 GPUs', 'n4': '4 GPUs', 'n4_numa': '4 GPUs\nNUMA-bound', 'n4_shm': '4 GPUs\nweights in shm'}
COLORS = {'n1': '#2a78d6', 'n2': '#eb6834', 'n4': '#1baf7a', 'n4_numa': '#eda100', 'n4_shm': '#e87ba4'}
STARTUP = [('imports', '#c3c2b7'), ('FCN3 load', '#8f8e87'), ('CorrDiff load', '#52514e'), ('stage to shm', '#e87ba4')]


def load(run_dir):
    configs = {}
    for name in ORDER:
        d = run_dir / name
        if not (d / 'wall.json').exists():
            continue
        wall = json.loads((d / 'wall.json').read_text())
        workers = [json.loads(p.read_text()) for p in sorted(d.glob('gpu*.json'))]
        members = [m for w in workers for m in w['members']]
        configs[name] = dict(
            wall=wall, workers=workers, members=members,
            e2e=wall['done'] - wall['launch'], stage=wall['staged'] - wall['launch'],
            compute=max(m['end'] for m in members) - min(w['compute_start'] for w in workers),
            startup=[dict(imports=w['imports_done'] - w['process_start'], fcn3=w['fcn3_load_sec'],
                          corrdiff=w['corrdiff_load_sec'], ready=w['compute_start'] - wall['launch']) for w in workers])
    return configs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = load(a.run)
    base = cfg['n1']
    n_members = len(base['members'])
    lines = [f'{n_members}-member FCN3 -> CorrDiff Taiwan ensemble, {base["workers"][0]["nsteps"]} FCN3 steps, '
             f'{base["workers"][0]["samples"]} CorrDiff samples per lead, {base["workers"][0]["gpu"]}', '',
             f'{"configuration":28s}{"end-to-end s":>13s}{"speedup":>9s}{"eff.":>7s}{"compute s":>11s}{"speedup":>9s}'
             f'{"eff.":>7s}{"ready s (max)":>15s}{"member s (med)":>16s}']
    for name, c in cfg.items():
        n = c['wall']['n_gpus']
        s_e2e, s_cmp = base['e2e'] / c['e2e'], base['compute'] / c['compute']
        member = np.median([m['end'] - m['start'] for m in c['members']])
        lines.append(f'{LABELS[name]:28s}{c["e2e"]:13.1f}{s_e2e:8.2f}x{s_e2e / n:7.2f}{c["compute"]:11.1f}{s_cmp:8.2f}x'
                     f'{s_cmp / n:7.2f}{max(s["ready"] for s in c["startup"]):15.1f}{member:16.2f}')
    lines += ['', 'Start-up per worker (mean / max seconds):']
    for name, c in cfg.items():
        st = c['startup']
        lines.append(f'  {LABELS[name]:28s} imports {np.mean([s["imports"] for s in st]):5.1f}/{max(s["imports"] for s in st):5.1f}'
                     f'  FCN3 load {np.mean([s["fcn3"] for s in st]):5.1f}/{max(s["fcn3"] for s in st):5.1f}'
                     f'  CorrDiff load {np.mean([s["corrdiff"] for s in st]):5.1f}/{max(s["corrdiff"] for s in st):5.1f}'
                     + (f'  staging {c["stage"]:.1f}' if c['stage'] > 0.5 else ''))
    lines += ['', 'Per-member medians (s): FCN3 total / CorrDiff total / copy+hash']
    for name, c in cfg.items():
        lines.append(f'  {LABELS[name]:28s} {np.median([m["fcn3_sec"] for m in c["members"]]):6.2f} / '
                     f'{np.median([m["corrdiff_sec"] for m in c["members"]]):6.2f} / '
                     f'{np.median([m["copy_hash_sec"] for m in c["members"]]):5.2f}')
    ref = {m['member']: m['sha256'] for m in base['members']}
    lines.append('')
    for name, c in cfg.items():
        same = sum(ref.get(m['member']) == m['sha256'] for m in c['members'])
        lines.append(f'bitwise identical to 1-GPU results: {LABELS[name]:28s} {same}/{len(c["members"])} members')
    text = '\n'.join(lines)
    print(text)
    (a.output / 'ensemble_scaling_summary.txt').write_text(text + '\n')
    plot(cfg, base, a.output / 'ensemble_scaling.png')


def plot(cfg, base, out):
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK_2, 'axes.labelcolor': INK_2, 'xtick.color': INK_2,
                         'ytick.color': INK_2, 'axes.titlecolor': INK, 'axes.titlesize': 10})
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0, 0])
    gpus = [1, 2, 4]
    ax.plot(gpus, gpus, ':', color=INK_2, lw=1, label='ideal')
    scaled = [k for k in ('n1', 'n2', 'n4') if k in cfg]
    ax.plot([cfg[k]['wall']['n_gpus'] for k in scaled], [base['compute'] / cfg[k]['compute'] for k in scaled],
            'o-', color=INK, lw=2, ms=6, label='compute phase')
    ax.plot([cfg[k]['wall']['n_gpus'] for k in scaled], [base['e2e'] / cfg[k]['e2e'] for k in scaled],
            's--', color=INK_2, lw=2, ms=6, label='end-to-end (incl. start-up)')
    for k in ('n4_numa', 'n4_shm'):
        if k in cfg:
            ax.scatter([4], [base['e2e'] / cfg[k]['e2e']], s=60, marker='D', color=COLORS[k], zorder=4,
                       edgecolor='white', label=f'end-to-end, {LABELS[k][8:]}')
    ax.set_xticks(gpus); ax.set_xlabel('GPUs (one worker each)'); ax.set_ylabel('speedup vs 1 GPU')
    ax.set_ylim(0, 4.4); ax.set_title('(a) Strong scaling, 16 members', loc='left')
    ax.legend(frameon=False, fontsize=7, loc='upper left')

    ax = fig.add_subplot(gs[0, 1])
    names = list(cfg)
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for (label, color), key in zip(STARTUP, ('imports', 'fcn3', 'corrdiff', 'stage')):
        if key == 'stage':
            vals = np.array([cfg[n]['stage'] for n in names])
        else:
            vals = np.array([np.mean([s[key] for s in cfg[n]['startup']]) for n in names])
        if vals.max() > 0.5:
            ax.bar(x, vals, 0.6, bottom=bottom, color=color, label=label, edgecolor='white', lw=1)
            bottom += vals
    ax.set_xticks(x, [SHORT[n] for n in names], fontsize=7)
    ax.set_ylim(0, bottom.max() * 1.25)
    ax.set_ylabel('seconds (mean per worker)'); ax.set_title('(b) Start-up before computing', loc='left')
    ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper right')

    ax = fig.add_subplot(gs[0, 2])
    for i, n in enumerate(names):
        t = [m['end'] - m['start'] for m in cfg[n]['members']]
        ax.scatter(np.full(len(t), i) + np.linspace(-0.15, 0.15, len(t)), t, s=14, color=COLORS[n], alpha=0.8)
        ax.plot([i - 0.25, i + 0.25], [np.median(t)] * 2, color=INK, lw=2)
    first = max(w['members'][0]['end'] - w['members'][0]['start'] for n in names for w in cfg[n]['workers'])
    ax.annotate("each worker's first member\nincludes warm-up", (0, first), xytext=(14, -6),
                textcoords='offset points', fontsize=7, color=INK_2, va='top')
    ax.set_xticks(x, [SHORT[n] for n in names], fontsize=7)
    ax.set_ylabel('seconds per member'); ax.set_ylim(0, first * 1.15)
    ax.set_title('(c) Time per member (bar = median)', loc='left')

    ax = fig.add_subplot(gs[1, :])
    positions, labels, row = [], [], 0.0
    for n in [k for k in ('n1', 'n4', 'n4_shm') if k in cfg]:
        c = cfg[n]; t0 = c['wall']['launch']
        for w in c['workers']:
            ax.barh(row, w['compute_start'] - t0, left=0, color='#c3c2b7', height=0.7)
            for j, m in enumerate(w['members']):
                ax.barh(row, m['end'] - m['start'], left=m['start'] - t0, color=COLORS[n], height=0.7,
                        alpha=0.95 if j % 2 == 0 else 0.6, edgecolor='white', lw=0.5)
            positions.append(row)
            labels.append(f'{LABELS[n]}  GPU{w["cuda_visible_devices"]}')
            row += 1
        row += 0.5   # gap between configurations
    ax.set_yticks(positions, labels, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel('seconds since launch')
    ax.set_title('(d) GPU timelines: grey = Python start-up and model loading, coloured = one member each', loc='left')
    for axis in fig.axes:
        axis.grid(axis='x' if axis is ax else 'y', color=GRID, lw=0.6); axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('FCN3 -> CorrDiff Taiwan ensemble on one node (4x A100-SXM4-80GB, NVLink; workers independent)',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
