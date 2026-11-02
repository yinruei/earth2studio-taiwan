"""Summarize and plot train_scaling_job.sh: DDP weak scaling (FP32/BF16), NCCL cost and the training curve.

The persistence baseline (predict the next 6-hour field as the current one) is computed on the same held-out
members and normalization as the training script, so its MSE is directly comparable with the validation loss.
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
PREC = {'fp32': '#2a78d6', 'bf16': '#eb6834'}   # categorical slots 1, 2


def persistence_mse(data_dir, val_members):
    arrays = {int(f.stem.split('_')[1]): np.load(f) for f in sorted(Path(data_dir).glob('member_*.npy'))}
    train = np.concatenate([a for m, a in arrays.items() if m not in val_members])
    mean, std = train.mean(axis=(0, 2, 3), keepdims=True), train.std(axis=(0, 2, 3), keepdims=True)
    err = [((arrays[m][1:] - arrays[m][:-1]) / std) ** 2 for m in arrays if m in val_members]
    return float(np.concatenate(err).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    bench = {(prec, n): json.loads((a.run / f'bench_n{n}_{prec}.json').read_text())
             for prec in PREC for n in (1, 2, 4)}
    train = json.loads((a.run / 'train_n4_bf16.json').read_text())
    prof = json.loads((a.run / 'profile_n4_bf16.json').read_text())
    rows = list(csv.DictReader(open(a.run / 'ddp_n4_bf16_cuda_gpu_kern_sum.csv')))
    total_ns = sum(float(r['Total Time (ns)']) for r in rows)
    nccl = [r for r in rows if 'nccl' in r['Name'].lower()]
    nccl_ns = sum(float(r['Total Time (ns)']) for r in nccl)
    steps_profiled = 20
    world = 4
    nccl_ms_per_step = nccl_ns / 1e6 / (steps_profiled * world)
    exposed_ms = (bench[('bf16', 4)]['step_median_sec'] - bench[('bf16', 1)]['step_median_sec']) * 1e3
    persist = persistence_mse(a.data, set(range(28, 32)))
    b1 = bench[('fp32', 1)]

    lines = [f'PhysicsNeMo AFNO ({b1["params"] / 1e6:.1f} M parameters, embed {b1["embed_dim"]}, depth {b1["depth"]}, '
             f'patch {b1["patch"]}), {b1["train_pairs"]} training pairs on a 64x64x8 regional grid, '
             f'per-GPU batch {b1["per_gpu_batch"]}, {b1["gpu"]}, torch {b1["torch"]}, NCCL {b1["nccl"]}', '',
             f'{"precision":10s}{"GPUs":>5s}{"step ms":>9s}{"samples/s":>11s}{"throughput x":>14s}{"weak-scaling eff.":>19s}{"peak GiB":>10s}']
    for prec in PREC:
        for n in (1, 2, 4):
            d, base = bench[(prec, n)], bench[(prec, 1)]
            lines.append(f'{prec:10s}{n:5d}{d["step_median_sec"] * 1e3:9.1f}{d["samples_per_sec"]:11.0f}'
                         f'{d["samples_per_sec"] / base["samples_per_sec"]:13.2f}x'
                         f'{base["step_median_sec"] / d["step_median_sec"]:19.2f}{d["peak_allocated_gib"]:10.2f}')
    lines += ['', f'BF16 vs FP32 on 1 GPU: {b1["step_median_sec"] / bench[("bf16", 1)]["step_median_sec"]:.2f}x faster',
              f'Nsight, 4-GPU BF16 ({steps_profiled} steps): NCCL all-reduce = {100 * nccl_ns / total_ns:.1f}% of GPU kernel '
              f'time, {nccl_ms_per_step:.1f} ms per GPU per step; '
              + ', '.join(f'{r["Name"].split("(")[0]} x{r["Instances"]}' for r in nccl),
              f'Step time added by going from 1 to 4 GPUs (BF16): {exposed_ms:.1f} ms, so about '
              f'{max(nccl_ms_per_step - exposed_ms, 0):.1f} ms of the all-reduce overlaps with the backward pass',
              f'(profiled step {prof["step_median_sec"] * 1e3:.1f} ms includes profiler overhead)', '',
              f'Training run, 4 GPUs BF16, {train["log"][-1]["step"]} steps: validation MSE '
              f'{train["log"][0]["val_loss"]:.4f} -> {train["log"][-1]["val_loss"]:.4f} (normalized units); '
              f'persistence baseline {persist:.4f}']
    text = '\n'.join(lines)
    print(text)
    (a.output / 'train_scaling_summary.txt').write_text(text + '\n')

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK_2, 'axes.labelcolor': INK_2, 'xtick.color': INK_2,
                         'ytick.color': INK_2, 'axes.titlecolor': INK, 'axes.titlesize': 10})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))
    gpus = [1, 2, 4]
    for prec, color in PREC.items():
        thr = [bench[(prec, n)]['samples_per_sec'] for n in gpus]
        ax1.plot(gpus, [thr[0] * n for n in gpus], ':', color=color, lw=1)
        eff = bench[(prec, 1)]['step_median_sec'] / bench[(prec, 4)]['step_median_sec']
        ax1.plot(gpus, thr, 'o-', color=color, lw=2, ms=6,
                 label=f'{prec.upper()}: {thr[-1]:.0f} samples/s on 4 GPUs, efficiency {eff:.2f}')
    ax1.set_xticks(gpus); ax1.set_xlabel('GPUs (per-GPU batch 16)'); ax1.set_ylabel('training samples / s')
    ax1.set_ylim(bottom=0); ax1.set_title('(a) DDP weak scaling (dotted: ideal)', loc='left')
    ax1.legend(frameon=False, fontsize=8, loc='upper left')

    labels = ['FP32\n1 GPU', 'FP32\n4 GPUs', 'BF16\n1 GPU', 'BF16\n4 GPUs']
    vals = [bench[(p_, n)]['step_median_sec'] * 1e3 for p_ in PREC for n in (1, 4)]
    bars = ax2.bar(labels, vals, color=[PREC['fp32']] * 2 + [PREC['bf16']] * 2, width=0.6, edgecolor='white', lw=2)
    ax2.bar_label(bars, fmt='%.1f ms', fontsize=7, color=INK_2, padding=2)
    ax2.annotate(f'NCCL all-reduce {nccl_ms_per_step:.1f} ms/step,\n{exposed_ms:.1f} ms of it not hidden',
                 (3, vals[3]), xytext=(-10, 26), textcoords='offset points', ha='right', fontsize=7, color=INK_2)
    ax2.set_ylim(0, max(vals) * 1.3); ax2.set_ylabel('ms per training step')
    ax2.set_title('(b) Step time: compute vs communication', loc='left')

    steps = [e['step'] for e in train['log']]
    ax3.plot(steps, [e['train_loss'] for e in train['log']], '-', color=INK_2, lw=1.5, label='train (mean over GPUs)')
    ax3.plot(steps, [e['val_loss'] for e in train['log']], '-', color=PREC['bf16'], lw=2, label='validation (held-out members)')
    ax3.axhline(persist, color=INK, ls='--', lw=1)
    ax3.annotate(f'persistence {persist:.3f}', (steps[-1], persist), xytext=(0, 4), textcoords='offset points',
                 ha='right', fontsize=7, color=INK)
    ax3.set_yscale('log'); ax3.set_xlabel('training step (4 GPUs, BF16)'); ax3.set_ylabel('MSE (normalized)')
    ax3.set_title('(c) Training curve', loc='left'); ax3.legend(frameon=False, fontsize=8)
    for ax in (ax1, ax2, ax3):
        ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('PhysicsNeMo AFNO data-parallel training on 4x A100 (NVLink, NCCL); FCN3-generated regional data',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(a.output / 'train_scaling.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
