# NVIDIA driver 590.48.01 with P2P for 4090 and 5090

This allows using P2P on 4090 and 5090 GPUs with the 590.48.01 driver version.  
See https://github.com/tinygrad/open-gpu-kernel-modules (various branches) for more info.

## What this `-48g` branch adds

On top of the base P2P mod, this branch (`590.48.01-p2p-48g`) adds support for **48GB-modded
RTX 4090** cards, whose 32GB PCIe BAR1 is smaller than their 48GB framebuffer. Stock BAR1 P2P
requires BAR1 ≥ VRAM (it identity-maps the whole framebuffer), so it cannot reach allocations
above 32GB. New elements:

- **Method 3 — dynamic per-allocation BAR1 P2P** (`docs/48g-4090-p2p/`). Each peer-mapped
  allocation is mapped into a BAR1 window on demand; the GMMU translates the ≤32GB BAR1 VA to the
  allocation's real framebuffer page anywhere in the 48GB. This keeps **all 48GB VRAM usable at
  full PCIe Gen4 P2P bandwidth**, entirely in the kernel driver — no userspace / `libcuda` patch,
  no `OverrideFbSize` cap. The implementation is shared with the `595.71.05-p2p-48g` branch and was
  validated on hardware there (8× RTX 4090 48G): 22.7 GB/s peer copy, ~20 GB/s NCCL all-reduce
  busbw, and 1.7× (2-socket Intel) up to 11× (AMD Rome) DDP step-time speedup. NCCL needs
  `NCCL_P2P_LEVEL=SYS`. Details:
  [`method3-dynamic-bar1-p2p.md`](docs/48g-4090-p2p/method3-dynamic-bar1-p2p.md),
  [`method3-test-results.md`](docs/48g-4090-p2p/method3-test-results.md).
- **Offline DKMS package** (`packaging/dkms/`) — builds `nvidia-open-p2p-dkms`, a drop-in
  replacement for the CUDA-repo `nvidia-kernel-open-dkms` that vendors this patched source and
  builds fully offline on the target. See [`packaging/dkms/README.md`](packaging/dkms/README.md).

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
