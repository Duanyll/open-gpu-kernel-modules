# NVIDIA driver 590.48.01 with P2P for 4090 and 5090

This allows using P2P on 4090 and 5090 GPUs with the 590.48.01 driver version.  
See https://github.com/tinygrad/open-gpu-kernel-modules (various branches) for more info.

## What this `-48g` branch adds

On top of the base P2P mod, this branch (`590.48.01-p2p-48g`) adds support for 48 GB-modded
RTX 4090s, whose 32 GB BAR1 is smaller than their 48 GB framebuffer. Stock BAR1 P2P
identity-maps the whole framebuffer, so it cannot reach allocations above 32 GB.

- **Method 3: dynamic per-allocation BAR1 P2P** ([`docs/48g-4090-p2p/`](docs/48g-4090-p2p/)).
  Each peer-mapped allocation gets its own BAR1 window, translated by the GMMU to the
  allocation's real framebuffer pages. With the cuMem peer-access path that NCCL and
  PyTorch use by default (`P2P/CUMEM`), only communication buffers take BAR1 space and
  the full 48 GB stays usable. Kernel-only: no `libcuda` patch, no `OverrideFbSize` cap.
  With `iommu=pt` and ACS off: ~26 GB/s peer copy, ~25 GB/s NCCL all-reduce busbw. Set
  `NCCL_P2P_LEVEL=SYS`. Legacy whole-device peer access (`cudaDeviceEnablePeerAccess`,
  e.g. cuda-samples `simpleP2P`) is capped at the 32 GB aperture. Details:
  [usable VRAM](docs/48g-4090-p2p/method3-usable-vram-cumem-vs-legacy.md),
  [design](docs/48g-4090-p2p/method3-dynamic-bar1-p2p.md),
  [test results](docs/48g-4090-p2p/method3-test-results.md).
- **Offline DKMS package** ([`packaging/dkms/`](packaging/dkms/README.md)): builds
  `nvidia-open-p2p-dkms`, a drop-in replacement for the CUDA-repo `nvidia-kernel-open-dkms`
  that vendors this patched source and builds offline on the target.

The [`615.71.09-p2p-48g`](https://github.com/Duanyll/open-gpu-kernel-modules/tree/615.71.09-p2p-48g)
branch adds GeForce GPUDirect RDMA and is what we run in production.

### Tuning tips

- **GEMM slows down while FSDP overlaps compute with communication?** Lower the NCCL
  channel count. Within one node, `NCCL_MAX_NCHANNELS=1` still saturates PCIe 4.0 and
  minimizes the impact on GEMM. With the default channel count, communication competes
  with compute for some memory-access resource on AD102.

The rest of this README describes the base P2P mod and is unchanged.

## How it works

This modifies the kernel driver to force enable BAR1 P2P mode on GPUs not intended to use it.  
Then, the transfers are done by directly writing to the other GPU physical addresses over DMA.  

IOMMU virtualization must be disabled to use the patch, or transfers will fail.  
Note that this is very dangerous if you run untrusted software or devices.

## How to use

1) Enable DMA passthrough mode for IOMMU:  
    1) Edit `/etc/default/grub`
    2) Add `amd_iommu=on iommu=pt` to `GRUB_CMDLINE_LINUX_DEFAULT`
    3) Run `sudo update-grub`
2) Install https://www.nvidia.com/en-us/drivers/details/259267/
3) Run `./install.sh` in this repo
4) Reboot the server

## Potential issues

- On some systems you might additionally need to disable ACS.
- On some systems resizable BAR might be unavailable.
- 4090s come up with large BAR by default, but 5090s don't.
