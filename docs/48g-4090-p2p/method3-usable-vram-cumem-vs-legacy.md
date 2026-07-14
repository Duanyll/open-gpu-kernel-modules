# Method 3 — how much VRAM is usable *with* P2P: cuMem vs legacy peer access

**TL;DR.** On a 48 GB-modded RTX 4090 (32 GB BAR1) running this fork's Method 3, whether you
can use **all 48 GB while P2P is active** depends entirely on *which peer-access API the
application uses*:

| Peer-access path | Who uses it | Usable VRAM with P2P | >32 GB data |
| --- | --- | --- | --- |
| **cuMem** (`cuMemSetAccess`, `NCCL_CUMEM_ENABLE=1`) | modern NCCL (≥2.19, default), PyTorch ≥2.8, most current DL frameworks | **all ~47 GB** | **correct** |
| **legacy** (`cudaDeviceEnablePeerAccess`) | `cuda-samples/simpleP2P`, `p2pBandwidthLatencyTest`, older/hand-written CUDA | **~32 GB (BAR1 size)** | rejected / corrupts |

So "all 48 GB usable at full P2P bandwidth" is **true for cuMem-based frameworks** (the real
training/inference use case) and **false for legacy whole-device peer-access apps**. This is not
a driver bug that can be papered over — it is inherent to what each API asks the driver to do.

## Why the split exists

The driver cannot know, at allocation time, *which* buffers a peer GPU will need to read. The
two peer-access APIs resolve that ambiguity in opposite ways:

- **`cudaDeviceEnablePeerAccess(peer)`** opens the peer's **entire device** for access. Method 3
  must therefore make *every* allocation on that GPU reachable through the 32 GB BAR1 window. It
  eagerly maps allocations into BAR1 as they are created, so the process's total device memory is
  bounded by BAR1 (~32 GB), and any single allocation larger than 32 GB can never be mapped. When
  the budget is hit the driver logs, e.g.:

  ```
  NVRM: _nvGpuOpsDynBar1Create: METHOD3: dynamic BAR1 P2P budget exceeded on GPU1:
        mapped 0x0 + 0x880000000 (=34 GiB) + reserve 0x4000000 > BAR1 0x800000000 (=32 GiB)
  ```

  (a single 34 GiB peer-map request rejected against the 32 GiB BAR1). Writes that *do* land above
  32 GB are not remapped and corrupt.

- **cuMem (`cuMemCreate` + `cuMemSetAccess`)** grants peer access to **specific allocations only**.
  Modern NCCL allocates its own communication buffers this way and calls `cuMemSetAccess` on just
  those. Only the comm buffers (tens–hundreds of MB) enter the 32 GB BAR1; the model/activation
  memory is never peer-mapped and can occupy the full framebuffer. This is what makes Method 3
  deliver "all 48 GB usable with P2P" in practice.

## Hardware evidence (4× RTX 4090 48G, driver 595.71.05, `iommu=pt`, ACS off)

Method 3 confirmed active in `dmesg` (`_nvGpuOpsDynBar1Create: METHOD3 …`).

1. **Max device allocation, no P2P:** 46.75 GiB (torch caching allocator, 256 MiB chunks) — the
   full 48 GB is allocatable when peer access is never enabled.
2. **Max device allocation, legacy `cudaDeviceEnablePeerAccess` enabled:** **31.75 GiB** — capped
   at BAR1 (`maxp2p` test). The highest chunk that verified sat at physical ~31.5 GiB (below 32 GB).
3. **cuMem path — the decisive test** (`dist_p2p_bigmem.py`): each of 2 ranks holds a **38.0 GiB**
   resident tensor (well above 32 GB), then runs NCCL `all_reduce`:
   ```
   NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM
   rank0: resident model = 38.0 GiB — all_reduce OK  correct=True  peak_resident=39.4 GiB
   ```
   38 GB resident **and** correct P2P collective coexist, over the real `P2P/CUMEM` transport.

Bandwidth (independent of the above) is unaffected: `p2pcheck` 26.3 GB/s per pair, NCCL 4-GPU
`all_reduce` busbw ~24.8 GB/s, data-integrity clean — provided **ACS is off and `iommu=pt`**
(see below).

## Two operational prerequisites (easy to get wrong)

1. **cuMem must be on.** It is the default in NCCL ≥2.19 (`NCCL_CUMEM_ENABLE=1`) and in PyTorch
   ≥2.8. If you force `NCCL_CUMEM_ENABLE=0`, or use a legacy-peer-access app, you fall back to the
   32 GB cap. Verify the transport shows `via P2P/CUMEM` (not `P2P/direct`) under `NCCL_DEBUG=INFO`.
2. **ACS off + `iommu=pt`.** With PCIe ACS redirect enabled (common default after a reboot on
   un-managed hosts), P2P stays *correct* but bandwidth collapses (~3–9 GB/s) because traffic is
   bounced through the root complex. `iommu=pt` + ACS off restores full ~26 GB/s.

## Bottom line for the "48 GB usable" claim

- ✅ **Training/inference (PyTorch, modern NCCL): all ~47 GB usable with P2P, verified correct.**
- ⚠️ **Legacy `cudaDeviceEnablePeerAccess` apps (incl. `simpleP2P`): capped at ~32 GB;** don't
  claim those pass. If someone reproduces with `simpleP2P` they will still see the 32 GB wall —
  point them at a cuMem/NCCL workload instead.
