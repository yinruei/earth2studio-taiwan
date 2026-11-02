"""Figure and summary table for corrdiff_bench.py: sample scaling, configuration speed and deviation.

Deviation is the per-sample RMSE against the default configuration (FP16, 8 Euler steps) with the same
seed; it is shown relative to the seed-to-seed difference (same configuration, seed + 1), which is the
spread the generative model produces anyway.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
VARS = {'mrr': '#2a78d6', 't2m': '#eb6834', 'u10m': '#1baf7a', 'v10m': '#eda100'}   # categorical slots 1-4
CONFIGS = [('compiled', 'torch.compile (FP16)'), ('base', 'default: FP16, 8 steps'), ('steps4', '4 steps'),
           ('steps12', '12 steps'), ('steps18', '18 steps'), ('fp32_tf32', 'FP32 (TF32 convs)'),
           ('fp32_strict', 'FP32 strict (no TF32)')]


def kernel_shares(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    total = sum(float(r['Total Time (ns)']) for r in rows)
    groups = {'elementwise / norm / activation': 0.0, 'convolution (tensor core)': 0.0, 'other': 0.0}
    for r in rows:
        name, t = r['Name'], float(r['Total Time (ns)'])
        if any(k in name for k in ('elementwise', 'reduce_kernel', 'silu', 'CatArray', 'Functor')):
            groups['elementwise / norm / activation'] += t
        elif any(k in name for k in ('cutlass', 'xmma', 'conv', 'grad2d', 'implicit_gemm')):
            groups['convolution (tensor core)'] += t
        else:
            groups['other'] += t
    launches = sum(int(r['Instances']) for r in rows)
    return {k: v / total for k, v in groups.items()}, launches, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='corrdiff_bench.py output directory')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    d = json.loads((a.run / 'corrdiff_bench.json').read_text())
    res, acc = d['results'], d['accuracy']
    samples = sorted((r['config']['samples'], r) for r in res.values()
                     if r['config']['steps'] == 8 and r['config']['precision'] == 'fp16'
                     and not r['config']['compile'] and r['config']['seed'] == d['meta']['seed'] and 'error' not in r)
    shares, launches, kernel_ns = kernel_shares(a.run / 'base_profile_cuda_gpu_kern_sum.csv')
    prof = json.loads((a.run / 'profile_run.json').read_text())

    lines = [f'CorrDiff Taiwan, input = FCN3 +{d["meta"]["lead_hours"]} h, {d["meta"]["gpu"]}, '
             f'torch {d["meta"]["torch"]}, {d["meta"]["repeats"]} timed calls per configuration', '',
             f'{"configuration":24s}{"median s":>10s}{"per sample s":>14s}{"first call s":>14s}{"peak GiB":>10s}'
             f'{"vs default":>12s}']
    base_t = res['base']['median_sec']
    for key, label in CONFIGS:
        r = res[key]
        lines.append(f'{label:24s}{r["median_sec"]:10.3f}{r["per_sample_median_sec"]:14.3f}{r["first_call_sec"]:14.1f}'
                     f'{r["peak_allocated_gib"]:10.2f}{base_t / r["median_sec"]:11.2f}x')
    lines += ['', 'Sample scaling (FP16, 8 steps, one batch):',
              f'{"samples":>8s}{"median s":>10s}{"per sample s":>14s}{"peak GiB":>10s}']
    for n, r in samples:
        lines.append(f'{n:8d}{r["median_sec"]:10.3f}{r["per_sample_median_sec"]:14.3f}{r["peak_allocated_gib"]:10.2f}')
    lines += ['', 'Per-sample RMSE vs default (same seed); seed2 = default with seed + 1 (generative spread):',
              f'{"configuration":24s}' + ''.join(f'{v:>10s}' for v in VARS)]
    for key, label in [('seed2', 'seed + 1')] + [c for c in CONFIGS if c[0] != 'base']:
        lines.append(f'{label:24s}' + ''.join(f'{acc[key][v]["per_sample_rmse"]:10.3f}' for v in VARS))
    lines += ['', f'Nsight Systems, default config, 2 calls: {launches} kernel launches, '
              f'{kernel_ns / 1e6:.0f} ms GPU kernel time; profiled call {prof["median_sec"]:.2f} s',
              *[f'  {k}: {v * 100:.0f}% of GPU kernel time' for k, v in shares.items()]]
    text = '\n'.join(lines)
    print(text)
    (a.output / 'corrdiff_bench_summary.txt').write_text(text + '\n')

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK_2, 'axes.labelcolor': INK_2, 'xtick.color': INK_2,
                         'ytick.color': INK_2, 'axes.titlecolor': INK, 'axes.titlesize': 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    (ax1, ax2), (ax3, ax4) = axes
    ns = [n for n, _ in samples]
    ax1.plot(ns, [r['per_sample_median_sec'] for _, r in samples], 'o-', color=INK, lw=2, ms=6)
    for n, r in samples:
        ax1.annotate(f'{r["per_sample_median_sec"]:.2f}', (n, r['per_sample_median_sec']), xytext=(0, 7),
                     textcoords='offset points', ha='center', fontsize=7, color=INK_2)
    ax1.set_xscale('log', base=2); ax1.set_xticks(ns, [str(n) for n in ns])
    ax1.set_ylim(0, max(r['per_sample_median_sec'] for _, r in samples) * 1.3)
    ax1.set_xlabel('samples per call (one batch)'); ax1.set_ylabel('seconds per sample')
    ax1.set_title('(a) Time per downscaled sample (FP16, 8 steps)', loc='left')

    ax2.plot(ns, [r['peak_allocated_gib'] for _, r in samples], 'o-', color=INK, lw=2, ms=6)
    ax2.axhline(80, color=INK_2, ls=':', lw=1)
    ax2.annotate('A100 80 GB', (ns[0], 80), xytext=(2, -11), textcoords='offset points', fontsize=7, color=INK_2)
    ax2.set_xscale('log', base=2); ax2.set_xticks(ns, [str(n) for n in ns]); ax2.set_ylim(0, 88)
    ax2.set_xlabel('samples per call (one batch)'); ax2.set_ylabel('GiB')
    ax2.set_title('(b) Peak PyTorch allocated memory', loc='left')

    labels = [lab for _, lab in CONFIGS][::-1]
    times = [res[k]['median_sec'] for k, _ in CONFIGS][::-1]
    bars = ax3.barh(labels, times, color=[INK if k == 'base' else INK_2 for k, _ in CONFIGS][::-1], height=0.6)
    ax3.bar_label(bars, labels=[f'{t:.2f} s  ({base_t / t:.2f}x)' for t in times], padding=3, fontsize=7, color=INK_2)
    ax3.set_xlim(0, max(times) * 1.3); ax3.set_xlabel('seconds per call (4 samples)')
    ax3.set_title('(c) Speed by configuration', loc='left')

    keys = [k for k, _ in CONFIGS if k != 'base']
    x = np.arange(len(keys)); w = 0.2
    for j, (v, color) in enumerate(VARS.items()):
        ratio = [acc[k][v]['per_sample_rmse'] / acc['seed2'][v]['per_sample_rmse'] for k in keys]
        ax4.bar(x + (j - 1.5) * w, ratio, w * 0.9, color=color, label=v)
    ax4.axhline(1, color=INK, lw=1, ls='--')
    ax4.annotate('= seed-to-seed difference', (len(keys) - 0.5, 1), xytext=(0, 4), textcoords='offset points',
                 ha='right', fontsize=7, color=INK)
    ax4.set_xticks(x, [dict(CONFIGS)[k].replace(' (', '\n(') for k in keys], fontsize=7)
    ax4.set_ylabel('RMSE vs default / seed-to-seed RMSE')
    ax4.set_title('(d) Deviation from default, relative to generative spread', loc='left')
    ax4.legend(frameon=False, fontsize=8, ncol=4, loc='upper left')
    ax4.set_ylim(0, 1.25)
    for ax in axes.flat:
        ax.grid(axis='x' if ax is ax3 else 'y', color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'CorrDiff Taiwan inference on 1x A100 (input: FCN3 +{d["meta"]["lead_hours"]} h, Typhoon Kong-rey)',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.output / 'corrdiff_bench.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
