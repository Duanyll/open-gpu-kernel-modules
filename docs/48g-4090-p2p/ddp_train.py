#!/usr/bin/env python3
# Minimal DDP training smoke/perf test for BAR1 P2P validation.
# Synthetic large-MLP workload so the gradient all-reduce moves real data.
# Launch: torchrun --standalone --nproc_per_node=N ddp_train.py [--hidden 8192] [--layers 6] [--steps 60]
import os, time, argparse
import torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=8192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    layers = []
    for _ in range(args.layers):
        layers += [nn.Linear(args.hidden, args.hidden), nn.ReLU()]
    model = nn.Sequential(*layers).to(dev)
    ddp = DDP(model, device_ids=[local])
    opt = torch.optim.SGD(ddp.parameters(), lr=1e-3)
    nparam = sum(p.numel() for p in model.parameters())
    grad_bytes = nparam * 4

    x = torch.randn(args.batch, args.hidden, device=dev)
    tgt = torch.randn(args.batch, args.hidden, device=dev)
    loss_fn = nn.MSELoss()

    torch.cuda.synchronize(); dist.barrier()
    t0 = None; last_loss = None
    for step in range(args.steps):
        if step == args.warmup:
            torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        out = ddp(x)
        loss = loss_fn(out, tgt)
        loss.backward()      # triggers gradient all-reduce across GPUs
        opt.step()
        last_loss = loss.item()
    torch.cuda.synchronize(); dist.barrier()
    dt = time.time() - t0
    nsteps = args.steps - args.warmup
    if rank == 0:
        per_step = dt / nsteps
        # ring all-reduce busbw approximation: 2*(N-1)/N * grad_bytes / time
        busbw = 2 * (world - 1) / world * grad_bytes / per_step / 1e9
        print(f"[OK] world={world} hidden={args.hidden} layers={args.layers} "
              f"params={nparam/1e6:.0f}M grad={grad_bytes/1e6:.0f}MB "
              f"| {per_step*1e3:.1f} ms/step | {nsteps/dt:.1f} steps/s "
              f"| allreduce~{busbw:.1f} GB/s busbw | final_loss={last_loss:.4f}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
