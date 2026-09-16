# Hardware validation logs — 2026-07

Raw, unedited tool output from a P2P validation run on three machines. Hostnames and IPs are
scrubbed to generic role labels; nothing else is altered. Interpretation and the cuMem-vs-legacy
finding are in [`../method3-usable-vram-cumem-vs-legacy.md`](../method3-usable-vram-cumem-vs-legacy.md);
the original bring-up results are in [`../method3-test-results.md`](../method3-test-results.md).

## Machines

| dir | GPUs | driver | Method 3 | config at test |
| --- | --- | --- | --- | --- |
| `node6-modded48g` | 4× RTX 4090 **48G** | 595.71.05 (this fork) | yes (dmesg `METHOD3`) | `iommu` default, **ACS disabled at runtime** |
| `node32-modded48g-8x` | **8×** RTX 4090 **48G** (dual-socket) | 595.71.05 (this fork) | yes (dmesg `METHOD3`) | `iommu` default, **ACS disabled at runtime** |
| `node14-stock48g` | 4× RTX 4090 **48G** | 580.95.05 **stock** | — | control (P2P off) |
| `node13-modded24g` | 4× RTX 4090 **24G** | 590.48.01 (this fork) | yes (BAR1≥VRAM, static path) | `iommu=pt` + ACS off |

`node6` and `node14` are the same 48G hardware class → clean modded-vs-stock control. `node32` is
the 8-GPU dual-socket box (GPU0–3 / GPU4–7 on separate PCI domains), so its P2P spans cross-NUMA.

## Headline results

| metric | 48G modded 4× (node6) | 48G modded 8× (node32) | 48G stock (node14) | 24G modded (node13) |
| --- | --- | --- | --- | --- |
| `nvidia-smi topo -p2p` | **OK** all pairs | **OK** all 8×8 pairs | **CNS** all pairs | **OK** all pairs |
| `p2pBandwidthLatencyTest` P2P=Enabled | ~**26 GB/s** | ~**22.7 GB/s** (incl. cross-NUMA) | ~2–3 GB/s (no P2P) | ~26 GB/s |
| `p2pcheck` 256 MiB, verified | 26.3 GB/s, 0 bad | 22.7 GB/s, 0 bad | noP2P | 26.3 GB/s, 0 bad |
| NCCL all_reduce busbw (`P2P_LEVEL=SYS`) | **24.8 GB/s** | **20.4 GB/s** | 2.6 GB/s (SHM) | **25.1 GB/s** |
| NCCL all_reduce busbw (default) | 2.6 GB/s (SHM) | 9.9 GB/s (SHM) | 2.6 GB/s | 4.6 GB/s (SHM) |
| >32 GB usable w/ P2P (cuMem, `20-…`) | **PASS** — 38 GiB resident + `via P2P/CUMEM`, correct | **PASS** — same | n/a | n/a (24G) |
| legacy `cudaDeviceEnablePeerAccess` cap (`21-…`) | **31.75 GiB** (BAR1) | **31.75 GiB** (BAR1) | n/a | n/a |

The stock node is the control: identical 48G cards, P2P reported **CNS** and NCCL stays on SHM
(~2.6 GB/s) — exactly what the mod removes. `node6`/`node32` ship with ACS on + no `iommu=pt`
(see their `00-node-info.log`); ACS was disabled at runtime for the bandwidth runs, which lifts
P2P from ~3–9 GB/s to the numbers above.

## File legend (per machine)

| file | what |
| --- | --- |
| `00-node-info.log` | hostname, driver version, kernel cmdline, ACS SrcValid+ count |
| `01-nvidia-smi.log` | `nvidia-smi` |
| `02-nvidia-smi-memory.log` | `nvidia-smi -q -d MEMORY` (incl. 32 GB BAR1) |
| `03-topo.log` | `nvidia-smi topo -m` + `-p2p r/w` |
| `04-lspci-bar.log` | GPU BAR / Resizable BAR / link (the ChihayaK / 152334H `lspci` check) |
| `10-p2pBandwidthLatencyTest.log` | cuda-samples P2P bandwidth/latency matrices |
| `11-p2pcheck-256MiB.log` | per-pair `cudaMemcpyPeer` correctness + bandwidth (`p2pcheck.cu`) |
| `12-nccl-all_reduce.log` | `all_reduce_perf`, `NCCL_P2P_LEVEL=SYS` vs default |
| `13-nccl-collectives.log` | all_gather / reduce_scatter / alltoall, `P2P_LEVEL=SYS` |
| `20-bigmem-p2p-cumem.log` | 38 GiB resident + NCCL P2P (`dist_p2p_bigmem.py`) — the ">32 GB usable" proof |
| `21-maxp2p-legacy-32GB-cap.log` | legacy peer-access → ~32 GiB cap (`maxp2p.cu`) |

## Reproduce

Sources for the custom tests are one level up: `p2pcheck.cu`, `p2pbig.cu`, `maxp2p.cu`,
`dist_p2p_bigmem.py`. NCCL bandwidth needs **`iommu=pt` + PCIe ACS off**; the "48 GB usable"
result needs the **cuMem** path (`NCCL_CUMEM_ENABLE=1`, default in modern NCCL / PyTorch ≥2.8).
