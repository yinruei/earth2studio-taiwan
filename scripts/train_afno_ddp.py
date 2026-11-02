"""Distributed data-parallel training of a PhysicsNeMo AFNO 6-hour emulator on the FCN3 regional dataset.

Launch with torch.distributed.run, one process per GPU. The whole dataset (~100 MB) is placed on every GPU, so
steps time only compute and NCCL gradient all-reduce, not data loading. Per-GPU batch is fixed (weak scaling).
Benchmark mode times --steps steps after --warmup; training mode also logs validation loss on held-out members.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def load_pairs(data_dir, val_members, device):
    files = sorted(Path(data_dir).glob('member_*.npy'))
    arrays = {int(f.stem.split('_')[1]): np.load(f) for f in files}          # (steps + 1, C, H, W)
    train = np.concatenate([a for m, a in arrays.items() if m not in val_members])
    mean = train.mean(axis=(0, 2, 3), keepdims=True)
    std = train.std(axis=(0, 2, 3), keepdims=True)

    def pairs(members):
        xs, ys = [], []
        for m in members:
            a = (arrays[m] - mean) / std
            xs.append(a[:-1]); ys.append(a[1:])
        return (torch.tensor(np.concatenate(xs), dtype=torch.float32, device=device),
                torch.tensor(np.concatenate(ys), dtype=torch.float32, device=device))

    train_members = [m for m in arrays if m not in val_members]
    return pairs(train_members), pairs([m for m in arrays if m in val_members]), len(train_members)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    p.add_argument('--batch', type=int, default=16, help='per-GPU batch')
    p.add_argument('--embed-dim', type=int, default=512)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--patch', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--val-every', type=int, default=0, help='training mode: validation interval (0 = benchmark)')
    p.add_argument('--val-members', type=int, nargs='+', default=[28, 29, 30, 31])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--profile', action='store_true', help='wrap the timed steps in cudaProfilerStart/Stop')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.models.afno import AFNO
    DistributedManager.initialize()
    dm = DistributedManager()
    rank, world, device = dm.rank, dm.world_size, dm.device
    torch.manual_seed(a.seed)
    (xtr, ytr), (xva, yva), n_train_members = load_pairs(a.data, set(a.val_members), device)
    c, h, w = xtr.shape[1:]
    model = AFNO(inp_shape=[h, w], in_channels=c, out_channels=c, patch_size=[a.patch, a.patch],
                 embed_dim=a.embed_dim, depth=a.depth, num_blocks=8).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[dm.local_rank], gradient_as_bucket_view=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    autocast = torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.precision == 'bf16')
    gen = torch.Generator(device=device)
    gen.manual_seed(a.seed * 1000 + rank)                     # each rank draws a different batch

    def step():
        idx = torch.randint(0, xtr.shape[0], (a.batch,), device=device, generator=gen)
        with autocast:
            loss = torch.nn.functional.mse_loss(model(xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    @torch.no_grad()
    def validate():
        model.eval()
        with autocast:
            losses = [torch.nn.functional.mse_loss(model(xva[i:i + 32]), yva[i:i + 32], reduction='sum')
                      for i in range(0, xva.shape[0], 32)]
        model.train()
        return float(sum(losses) / yva.numel())

    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if world > 1:
        dist.barrier()
    if a.profile:
        torch.cuda.profiler.start()
    times, log = [], []
    for i in range(1, a.steps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if a.val_every and (i % a.val_every == 0 or i == a.steps):
            lt = loss.detach().clone()
            if world > 1:
                dist.all_reduce(lt, op=dist.ReduceOp.AVG)
            entry = dict(step=i, train_loss=float(lt), val_loss=validate() if rank == 0 else None)
            log.append(entry)
            if rank == 0:
                print(json.dumps(entry), flush=True)
    if a.profile:
        torch.cuda.profiler.stop()

    t = torch.tensor(times, device=device)
    gathered = [torch.zeros_like(t) for _ in range(world)] if world > 1 else [t]
    if world > 1:
        dist.all_gather(gathered, t)
    if rank == 0:
        slowest = torch.stack(gathered).max(dim=0).values.cpu().numpy()   # a step ends when the slowest rank ends
        med = float(np.median(slowest))
        result = dict(world_size=world, precision=a.precision, per_gpu_batch=a.batch, global_batch=a.batch * world,
                      params=n_params, embed_dim=a.embed_dim, depth=a.depth, patch=a.patch,
                      train_pairs=int(xtr.shape[0]), train_members=n_train_members, val_pairs=int(yva.shape[0]),
                      step_median_sec=med, step_p90_sec=float(np.percentile(slowest, 90)),
                      samples_per_sec=a.batch * world / med, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      final_loss=float(loss), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                      nccl='.'.join(map(str, torch.cuda.nccl.version())), log=log)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=1))
        print(json.dumps({k: v for k, v in result.items() if k != 'log'}), flush=True)
    DistributedManager.cleanup()   # also registered at exit; calling dist.destroy_process_group() here would run twice


if __name__ == '__main__':
    main()
