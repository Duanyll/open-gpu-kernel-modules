# Method 3 dynamic BAR1 P2P — test results (Platform A: 2-socket, 8× RTX 4090 48G)

> **Important follow-up (read first):** "all 48 GB usable with P2P" holds only via the **cuMem**
> peer-access path (modern NCCL / PyTorch ≥2.8, `NCCL_CUMEM_ENABLE=1`). Apps using the legacy
> `cudaDeviceEnablePeerAccess` API (e.g. `simpleP2P`) open the whole device and stay capped at
> ~32 GB. This was measured on hardware in July 2026 — see
> [`method3-usable-vram-cumem-vs-legacy.md`](method3-usable-vram-cumem-vs-legacy.md) and the raw
> logs in [`hw-logs-2026-07/`](hw-logs-2026-07/).

Test machine: Platform A — a 2-socket headless server node (dual Intel Xeon Silver 4416+).
- 8× RTX 4090 **48G** (leaked-VBIOS, 32GB BAR1), 49140 MiB each.
- Ubuntu 22.04.5, kernel 6.8.0-124-generic.
- Driver flavor: **nvidia-595-open** (open kernel modules), version **595.71.05** — exact match to branch `595.71.05-p2p-48g`.
- Topology: GPU0–3 on NUMA0 (PCI domain 0000), GPU4–7 on NUMA1 (PCI domain 0001).
  Intra-NUMA pairs = `NODE`; cross-NUMA = `SYS`. So the meaningful single-fabric P2P group is GPU0–3.
- NCCL: nccl-tests built against system NCCL 2.25.1 + CUDA 12.8 inside
  `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` container (`--gpus all --ipc=host --network host`).

## Baseline — STOCK open driver 595.71.05 (no patch)

`nvidia-smi topo -p2p r/w`: **all pairs CNS** (Chipset not supported) — zero P2P, as expected for stock GeForce.

`all_reduce_perf -b 8 -e 1G -f 2` (single process, `-g N`), peak busbw:

| config | algbw (GB/s) | busbw (GB/s) |
|---|---|---|
| 4-card NUMA0 (GPU0-3) | 11.2 | **16.8** |
| 8-card (all) | 8.1 | **14.2** |

(Sysmem/SHM-staged, no P2P.)

## Modified driver 595.71.05-p2p-48g (Method 3 dynamic BAR1 P2P)

Build/install: open modules built on host (gcc-11, 595.71.05, vermagic matches), stock `.ko`
backed up to `/root/nvidia-ko-backup-stock-595.71.05`, patched `.ko` installed + `depmod`.
In-place reload (`rmmod` stack → `modprobe nvidia`) works cleanly on this headless node
(refcounts 0), so dev iteration is ~5s, no reboot needed. SOL console on ttyS0 validated.

### Progressive findings

1. **Driver loads, P2P reported supported.** `nvidia-smi topo -p2p r/w` → **all pairs OK**
   (was all CNS on stock). The `kbusIsPcieBar1P2PMappingSupported_GH100` gate decouple works.
   Single-GPU CUDA compute works.

2. **BUG (fixed): multi-GPU CUDA init aborted.** CUDA eagerly creates NV503B P2P objects between
   all P2P-capable visible GPUs at init; the legacy path `p2p_api.c:615` called the *static*
   `kbusGetBar1P2PDmaInfo_HAL` unconditionally → `pPeerDmaMemDesc != NULL` assert
   (kern_bus_gh100.c:1680) → `NV_ERR_NOT_SUPPORTED` → init abort (`cudaErrorInitializationError`).
   FIX: guard that fetch with `kbusIsStaticBar1Enabled(local) && kbusIsStaticBar1Enabled(remote)`,
   leaving the default sentinel `dma_address=NV_U64_MAX, dma_size=0` (the UMD's "no BAR1 DMA info"
   state). After fix, multi-GPU init succeeds.

3. **Confirmed: plain CUDA-runtime P2P (`cudaDeviceEnablePeerAccess`/`cudaMemcpyPeer`, no UVM
   managed mem) DOES drive our dynamic path** (`nvGpuOpsBuildExternalAllocPtes` →
   `_nvGpuOpsDynBar1Create`). So Method 3 is on NCCL's hot path.

4. **BUG (FIXED): dynamic IOMMU map failed.** `_nvGpuOpsDynBar1Create` built the BAR1-window
   memdesc owned by `pMappingGpu` (local GPU). `osIovaMap` resolves the peer as the memdesc
   owner and checks `IS_FB_OFFSET(owner, windowPhys)`; with the local GPU as owner the remote
   BAR1 address doesn't match → falls to the `pPriv==NULL` branch → `NV_ERR_INVALID_STATE`,
   never reaching the cross-device `nv_dma_map_peer()` path. FIX (one token): make the window
   memdesc owned by `pRemoteGpu` (the FB owner), exactly as the static path
   (`kbusEnableStaticBar1Mapping_TU102` + `_kbusCreateStaticBar1IOMMUMapping`) does. IOMMU map
   target + readback still use `pMappingGpu`'s iovaspace (correct). `nv_gpu_ops.c:3794`.

### P2P correctness + bandwidth (custom `p2pcheck.cu`, `cudaMemcpyPeer`, 256 MB, verified)

After the two fixes (reboot into build3), **every ordered GPU pair passes data-integrity
verification at full PCIe Gen4 x16 P2P bandwidth**:

| config | result |
|---|---|
| GPU0–3 (single NUMA) | all 12 pairs **22.7 GB/s**, correctness PASS |
| all 8 GPUs (incl. cross-NUMA) | all 56 pairs **22.7 GB/s**, correctness PASS |

Cross-NUMA pairs (GPU0–3 ↔ GPU4–7, topo `SYS`) also work at full P2P bandwidth — the IOMMU /
root complex forwards the BAR1 P2P traffic across the CPU interconnect. dmesg clean (no NVRM
errors/asserts). ACS ReqRedir+/CmpltRedir+ is enabled but does not block P2P here (IOMMU on).

vs stock baseline (sysmem-staged ~16.8 GB/s for the 4-card all-reduce path) this is real P2P.

### NCCL (`all_reduce_perf`, single process, `-g N`, peak busbw at 1 GiB)

NCCL's default topology gating sets `intraNodeP2pSupport 0` for these GPUs (they sit at the
cross-PCIe-host-bridge "NODE" distance, above NCCL's default `NCCL_P2P_LEVEL`), so by default
NCCL falls back to `SHM/direct` (sysmem-staged) and gets ~stock numbers. Setting
**`NCCL_P2P_LEVEL=SYS`** flips it to `intraNodeP2pSupport 1` → `via P2P/direct pointer` → real
BAR1 P2P. (Env var only — no NCCL recompile / userspace code change.)

| config | stock (no P2P) | patched, default | patched, `NCCL_P2P_LEVEL=SYS` | correctness |
|---|---|---|---|---|
| 4-card NUMA0 | 16.8 | 15.2 (SHM) | **20.4** (P2P) | 0 wrong |
| 8-card (cross-NUMA) | 14.2 | — | **20.46** (P2P) | 0 wrong |

(busbw GB/s. `#wrong` column = 0 in every run → NCCL's built-in result validation passes.)

dmesg clean across all runs (no Xid / no `INSUFFICIENT_RESOURCES` from the BAR1 budget guard);
GPU memory fully released after tests → dynamic mapping lifecycle has no leak.

### PyTorch DDP training (`ddp_train.py`, synthetic 403M-param MLP, 1611 MB grad/step)

Real DDP training (gradient all-reduce each step), `torchrun --nproc_per_node=N`
(torch 2.8 bundles NCCL 2.27.3):

| config | ms/step | steps/s | all-reduce busbw | final loss |
|---|---|---|---|---|
| 4-card P2P (`NCCL_P2P_LEVEL=SYS`) | 134.7 | 7.4 | ~17.9 GB/s | 1.0042 |
| 8-card P2P (`NCCL_P2P_LEVEL=SYS`) | 153.4 | 6.5 | ~18.4 GB/s | 1.0005 |
| 8-card default (SHM, no P2P) | 259.5 | 3.9 | ~10.9 GB/s | 1.0007 |

→ P2P gives **~1.7× faster step time** for 8-card DDP, correct convergence, no hangs/errors.

## Verdict

**Method 3 works.** On the 48GB/32GB-BAR1 RTX 4090, driver-only dynamic per-allocation BAR1
P2P gives correct data at full PCIe Gen4 P2P bandwidth (22.7 GB/s peer copy; 20.4 GB/s NCCL
all-reduce busbw), for all 8 GPUs incl. cross-NUMA, **while keeping all 48GB VRAM usable**
(no `OverrideFbSize` cap). Two bugs were found + fixed on hardware:
1. `p2p_api.c` — guard the static `kbusGetBar1P2PDmaInfo` fetch behind `kbusIsStaticBar1Enabled`
   (else multi-GPU CUDA init aborts).
2. `nv_gpu_ops.c` `_nvGpuOpsDynBar1Create` — the BAR1-window memdesc must be owned by the FB
   owner (`pRemoteGpu`) so `osIovaMap` takes the cross-device `nv_dma_map_peer` path.

Operational note: NCCL needs `NCCL_P2P_LEVEL=SYS` to actually use P2P on this multi-host-bridge
box (otherwise it silently stays on SHM).

---

# Second platform — Platform B (AMD EPYC 7302 / Rome, 4× RTX 4090 48G)

Same driver (595.71.05 nvidia-595-open, kernel 6.8.0-124), 4 GPUs single socket / 1 NUMA node,
all pairs at NODE/PHB distance. IOMMU on (AMD-Vi), **ACS redirect OFF** (`ReqRedir-`/`CmpltRedir-`)
— no GRUB change needed. Desktop (gdm/Xorg) was disabled (`systemctl set-default multi-user.target`,
reversible) so X doesn't hold the GPUs. Driver built with the correctness hardening
(explicit `bDynBar1Mapped` flag; BAR1-exhaustion/OOM logging; mixed static/dynamic pair rejection).

Rome without P2P is *especially* bad (sysmem staging bottlenecked by the IO-die), so P2P helps far
more here than on the 2-socket Platform A:

| test | stock / SHM (no P2P) | patched + `NCCL_P2P_LEVEL=SYS` | speedup | correctness |
|---|---|---|---|---|
| `p2pcheck` peer copy (all 12 pairs) | n/a (CNS) | **26.3 GB/s** | — | ✅ all verified |
| NCCL all_reduce 4-card busbw | **4.1 GB/s** | **25.15 GB/s** | **6.1×** | 0 wrong |
| DDP 4-card (403M params, ms/step) | **1203.7** (2.0 GB/s AR) | **109.5** (22.1 GB/s AR) | **11×** | loss 0.9996 |

dmesg clean, no mapping leak. Confirms Method 3 + hardening works on a second, AMD-Rome platform,
where the P2P benefit is dramatic (6× collective, 11× DDP step time).

