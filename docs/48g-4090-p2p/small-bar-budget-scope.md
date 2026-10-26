# Scope the early dynamic BAR1 budget by destination

`_nvGpuOpsDynBar1Create` compares the early byte-budget estimate to
`pRemoteKernelBus`'s BAR1 size. The previous estimate used all bytes tracked
by the source subdevice, including mappings into other GPUs' BARs. Those
unrelated mappings must not consume the current destination's early budget.

This change sums existing entries with `pMap->pRemoteGpu == pRemoteGpu`.
It keeps the 64 MiB margin, the existing aggregate diagnostic counter and
the real `kbusMapFbAperture` allocation/failure path.

This is still a per-source/per-destination heuristic. It does not provide
global accounting across source subdevices/processes, deduplicate every
shared physical window, or guarantee a global 64 MiB reservation.

## Observation and validation scope

On a seven-GPU 48 GB RTX 4090 KVM guest with 256 MiB BAR1 apertures, our
580.173.02 Method 3 backport hit:

```text
mapped 0xc000000 + 0x1000000 + reserve 0x4000000 > BAR1 0x10000000
```

The accumulated 192 MiB included other destination GPUs. With the early
check scoped by destination, the same seven-GPU NCCL command passed. The
final 580 backport also passed all 42 directed CUDA read/write pairs at
five sizes for three rounds, and 2/4/7-GPU NCCL checks.

**Hardware testing was on 580.173.02, not this 590.48.01 branch.** This is a
minimal source-level port of the budget change for maintainer review. It
must not be described as a hardware-validated 590 driver.

Our 256 MiB environment additionally needed a separate peer-PTE alignment
workaround (64 KiB ceiling and unaligned-PTE rejection). That change is not
part of this PR: the budget fix alone is not a complete small-BAR bring-up
recipe. Full findings, exact tested patch, probes and numerical records:

https://github.com/xiaoyanzi191/rtx4090-small-bar-p2p

## Attribution / AI involvement

This builds on NVIDIA, tinygrad/geohot, aikitoria and Duanyll's existing
BAR1/Method 3 work; it does not claim authorship of that foundation.
The repository owner supplied research direction, goals and experiment
authorization. OpenAI Codex retrieved information, inspected source,
implemented/backported changes and scripts, ran experiments, analyzed
outputs and prepared this contribution. No independent third-party
replication or complete manual human code review is claimed.
