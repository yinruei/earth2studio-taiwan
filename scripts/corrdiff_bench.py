"""CorrDiff Taiwan inference benchmark: precision, diffusion steps, sample batching and torch.compile.

The CorrDiff input is produced once (GFS analysis -> FCN3 to --lead hours -> map_coords) and cached, so
every configuration downscales exactly the same field. Each configuration records the first call
(warm-up, includes compilation when enabled), --repeats timed calls, peak allocated memory and how
many times the diffusion network ran. Outputs of the 4-sample configurations are kept for accuracy
comparison against the default configuration (FP16 weights, 8 Euler steps), using the same seed.

FP32 configurations reload the weights from the checkpoint: Earth2Studio's loader sets use_fp16=True,
which converts the weights themselves, so flipping the flag back would leave FP16-rounded weights.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

NVTX = torch.cuda.nvtx


def build_input(a, device):
    """GFS analysis -> FCN3 rollout to a.lead -> CorrDiff input grid; cached on disk."""
    cache = a.output / f'corrdiff_input_lead{a.lead:02d}h.pt'
    if cache.exists():
        return torch.load(cache, weights_only=False)
    import earth2studio.models.px.fcn3  # noqa: F401  (registers optional deps)
    from earth2studio.data import DataArrayFile, fetch_data
    from earth2studio.models.auto import Package
    from earth2studio.models.px import FCN3
    from earth2studio.utils.coords import map_coords
    fcn3 = FCN3.load_model(Package(str(a.fcn3))).to(device)
    fcn3.set_rng(seed=a.seed)
    data = DataArrayFile(str(a.ic))
    ic = fcn3.input_coords()
    x, coords = fetch_data(data, time=data.da['time'].values[:1], variable=ic['variable'],
                           lead_time=ic['lead_time'], device=device)
    x, coords = map_coords(x, coords, ic)
    model = fcn3.create_iterator(x, coords)
    for _ in range(a.lead // 6 + 1):
        x, coords = next(model)
    payload = dict(x=x.cpu(), coords=coords, lead=int(coords['lead_time'][-1] / np.timedelta64(1, 'h')))
    torch.save(payload, cache)
    del fcn3, model, x
    torch.cuda.empty_cache()
    return payload


def load_corrdiff(path, device):
    from earth2studio.models.auto import Package
    from earth2studio.models.dx import CorrDiffTaiwan
    return CorrDiffTaiwan.load_model(Package(str(path)), device=device).to(device)


def fp32_networks(path, device):
    """Fresh FP32 copies of both networks, never cast to FP16."""
    from physicsnemo.diffusion.preconditioners.legacy import EDMPrecondSR
    from physicsnemo.models.diffusion_unets import UNet
    ckpt = Path(path) / 'corrdiff_inference_package' / 'checkpoints'
    nets = []
    for cls, name in ((EDMPrecondSR, 'diffusion.mdlus'), (UNet, 'regression.mdlus')):
        net = cls.from_checkpoint(str(ckpt / name), override_args={'use_apex_gn': False})
        net.use_fp16, net.profile_mode = False, False
        nets.append(net.eval().to(device).to(memory_format=torch.channels_last))
    return nets


class CallCounter:
    """Forward hooks that count network calls and mark them as NVTX ranges for Nsight Systems.

    Hooks are detached for torch.compile configurations, where they would cause graph breaks.
    """

    def __init__(self, label):
        self.label, self.n, self.handles = label, 0, []

    def attach(self, module):
        self.detach()
        self.handles = [module.register_forward_pre_hook(self._pre), module.register_forward_hook(self._post)]

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _pre(self, *_):
        self.n += 1
        NVTX.range_push(self.label)

    def _post(self, *_):
        NVTX.range_pop()


def run_config(corrdiff, x, coords, cfg, repeats, counters):
    from earth2studio.utils.coords import map_coords
    corrdiff.number_of_samples, corrdiff.number_of_steps = cfg['samples'], cfg['steps']
    corrdiff.seed = cfg['seed']
    xd, cdc = map_coords(x, coords, corrdiff.input_coords())

    def call():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        NVTX.range_push(f'corrdiff {cfg["name"]}')
        y, _ = corrdiff(xd, cdc)
        NVTX.range_pop()
        torch.cuda.synchronize()
        return y, time.perf_counter() - t0

    for c in counters.values():
        c.n = 0
    y, first = call()
    calls = {k: c.n for k, c in counters.items()}
    torch.cuda.reset_peak_memory_stats()
    times = [call()[1] for _ in range(repeats)]
    return y, dict(first_call_sec=first, times_sec=times, median_sec=float(np.median(times)),
                   per_sample_median_sec=float(np.median(times)) / cfg['samples'],
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, network_calls=calls)


def configs(seed, sample_list):
    base = dict(precision='fp16', steps=8, samples=4, seed=seed, compile=False)
    out = [dict(base, name='base')]
    out.append(dict(base, name='seed2', seed=seed + 1))            # stochastic noise floor
    out += [dict(base, name=f'steps{s}', steps=s) for s in (4, 12, 18)]
    out += [dict(base, name=f'samples{n}', samples=n) for n in sample_list if n != 4]
    out.append(dict(base, name='fp32_tf32', precision='fp32'))
    out.append(dict(base, name='fp32_strict', precision='fp32', tf32=False))
    out.append(dict(base, name='compiled', compile=True))          # last: compile wraps the networks
    return out


def accuracy(results, outputs):
    ref = outputs['base']
    variables = ['mrr', 't2m', 'u10m', 'v10m']
    table = {}
    for name, y in outputs.items():
        if name == 'base':
            continue
        d = y - ref
        table[name] = {v: dict(per_sample_rmse=float(np.sqrt((d[:, i] ** 2).mean())),
                               sample_mean_rmse=float(np.sqrt((d[:, i].mean(0) ** 2).mean())),
                               max_abs=float(np.abs(d[:, i]).max()))
                       for i, v in enumerate(variables)}
        wind = np.hypot(y[:, 2], y[:, 3]).max(axis=(1, 2))
        table[name]['max_wind_per_sample'] = [round(float(w), 2) for w in wind]
    table['base'] = dict(max_wind_per_sample=[round(float(w), 2) for w in np.hypot(ref[:, 2], ref[:, 3]).max(axis=(1, 2))],
                         spread_std={v: float(ref[:, i].std(0).mean()) for i, v in enumerate(variables)})
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ic', type=Path, required=True)
    p.add_argument('--fcn3', type=Path, required=True)
    p.add_argument('--corrdiff', type=Path, required=True)
    p.add_argument('--lead', type=int, default=30)
    p.add_argument('--seed', type=int, default=20241030)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--samples-list', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument('--only', nargs='+', help='run only these configuration names')
    p.add_argument('--profile', action='store_true',
                   help='Nsight Systems mode: run base, then capture 2 calls between cudaProfilerStart/Stop')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    payload = build_input(a, device)
    x = payload['x'].to(device)
    coords = payload['coords']
    corrdiff = load_corrdiff(a.corrdiff, device)
    fp16_nets = (corrdiff.residual_model, corrdiff.regression_model)
    counters = {'diffusion': CallCounter('diffusion net'), 'regression': CallCounter('regression net')}
    for c, net in zip(counters.values(), fp16_nets):
        c.attach(net)

    if a.profile:
        cfg = configs(a.seed, a.samples_list)[0]
        run_config(corrdiff, x, coords, cfg, 1, counters)
        torch.cuda.profiler.start()
        _, res = run_config(corrdiff, x, coords, cfg, 1, counters)
        torch.cuda.profiler.stop()
        (a.output / 'profile_run.json').write_text(json.dumps(res, indent=2))
        print(json.dumps(res))
        return

    results, outputs = {}, {}
    fp32_nets = None
    for cfg in configs(a.seed, a.samples_list):
        if a.only and cfg['name'] not in a.only and cfg['name'] != 'base':
            continue
        torch.backends.cudnn.allow_tf32 = cfg.get('tf32', True)
        torch.backends.cuda.matmul.allow_tf32 = cfg.get('tf32', True) and cfg['precision'] == 'fp32'
        if cfg['precision'] == 'fp32':
            fp32_nets = fp32_nets or fp32_networks(a.corrdiff, device)
            nets = fp32_nets
        else:
            nets = fp16_nets
        for c, net in zip(counters.values(), nets):
            c.detach() if cfg['compile'] else c.attach(net)
        if cfg['compile']:
            nets = (torch.compile(nets[0]), torch.compile(nets[1]))
        corrdiff.residual_model, corrdiff.regression_model = nets
        try:
            y, res = run_config(corrdiff, x, coords, cfg, a.repeats, counters)
        except torch.cuda.OutOfMemoryError:
            res = dict(error='CUDA out of memory')
            torch.cuda.empty_cache()
            y = None
        res.update(config=cfg, matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_tf32=torch.backends.cudnn.allow_tf32)
        results[cfg['name']] = res
        print(json.dumps({'name': cfg['name'], **{k: v for k, v in res.items() if k != 'config'}}), flush=True)
        if y is not None and cfg['samples'] == 4:
            arr = y.float().cpu().numpy().reshape(4, -1, *y.shape[-2:])
            outputs[cfg['name']] = arr
            np.save(a.output / f'out_{cfg["name"]}.npy', arr)
        del y
        torch.cuda.empty_cache()

    acc = accuracy(results, outputs) if 'base' in outputs else {}
    meta = dict(lead_hours=payload['lead'], gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                repeats=a.repeats, seed=a.seed, torchdynamo_disabled=os.environ.get('TORCHDYNAMO_DISABLE') == '1')
    (a.output / 'corrdiff_bench.json').write_text(json.dumps(dict(meta=meta, results=results, accuracy=acc), indent=2))
    print(json.dumps(acc, indent=1))


if __name__ == '__main__':
    main()
