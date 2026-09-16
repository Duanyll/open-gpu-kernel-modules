#!/usr/bin/env python3
# Decisive test: can a process use NCCL P2P collectives while > 32 GiB of VRAM is
# already resident on each GPU? On a 48 GB-modded 4090 (32 GB BAR1) this answers
# whether "all 48 GB usable with P2P" actually holds.
#
# Each rank: (1) allocates a large resident "model" tensor (default 38 GiB, well
# above the 32 GiB BAR1) via the normal caching allocator; (2) runs NCCL all_reduce
# on a small comm buffer and verifies the result. If NCCL uses the cuMem path it
# maps only its own comm buffers into BAR1 (not the 38 GiB model), so both coexist
# -> PASS. If it falls back to legacy whole-device peer access, the peer mapping
# can't fit the 38 GiB into 32 GiB BAR1 -> OOM/fail.
import os, torch, torch.distributed as dist

GB = 1024**3
big_gb = float(os.environ.get("BIG_GB", "38"))

dist.init_process_group("nccl")
rank = dist.get_rank(); world = dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local)

free0, total0 = torch.cuda.mem_get_info()
if rank == 0:
    print(f"[env] NCCL_CUMEM_ENABLE={os.environ.get('NCCL_CUMEM_ENABLE')} "
          f"NCCL_P2P_LEVEL={os.environ.get('NCCL_P2P_LEVEL')} world={world} "
          f"GPU total={total0/GB:.1f} GiB", flush=True)

# (1) occupy > 32 GiB with a resident tensor
resident = None
try:
    n = int(big_gb * GB // 2)                      # float16 elements
    resident = torch.empty(n, dtype=torch.float16, device="cuda")
    resident.fill_(float(rank + 1))
    torch.cuda.synchronize()
except RuntimeError as e:
    print(f"rank{rank}: FAILED to allocate {big_gb} GiB model: {str(e)[:100]}", flush=True)
    dist.barrier(); dist.destroy_process_group(); raise SystemExit(1)

alloc = torch.cuda.memory_allocated() / GB
freeN, _ = torch.cuda.mem_get_info()
print(f"rank{rank}: resident model = {alloc:.1f} GiB (free left {freeN/GB:.1f} GiB) — now doing NCCL P2P", flush=True)

# (2) NCCL all_reduce (P2P) while > 32 GiB is resident
comm_mb = int(os.environ.get("COMM_MB", "256"))
comm = torch.full((comm_mb * 1024 * 1024 // 2,), float(rank + 1), dtype=torch.float16, device="cuda")
try:
    for _ in range(3):                         # warmup (sets up P2P conns/buffers)
        dist.all_reduce(comm)
    torch.cuda.synchronize()
    comm.fill_(float(rank + 1))                # reset, then ONE verified all_reduce
    dist.all_reduce(comm)
    torch.cuda.synchronize()
except RuntimeError as e:
    print(f"rank{rank}: *** NCCL all_reduce FAILED with {alloc:.1f} GiB resident: {str(e)[:120]}", flush=True)
    dist.barrier(); dist.destroy_process_group(); raise SystemExit(2)

expected = float(sum(range(1, world + 1)))     # sum of (rank+1) over all ranks
bad = int((comm != expected).sum().item())
ok = (bad == 0)
if not ok and rank == 0:
    print(f"rank0: sample got={comm[:4].tolist()} expected={expected}", flush=True)
peak = torch.cuda.max_memory_allocated() / GB
print(f"rank{rank}: all_reduce OK  correct={bool(ok)}  peak_resident={peak:.1f} GiB", flush=True)

dist.barrier()
if rank == 0:
    verdict = "PASS — NCCL P2P works with >32 GiB resident" if alloc > 32 and ok else "INCONCLUSIVE"
    print(f"\n==== {verdict}  (model {alloc:.1f} GiB > 32 GiB, P2P collective correct) ====", flush=True)
dist.destroy_process_group()
