# NVIDIA driver 580.95.05 with P2P for 4090 and 5090

This allows using P2P on 4090 and 5090 GPUs with the 580.95.05 driver version.
See https://github.com/tinygrad/open-gpu-kernel-modules (various branches) for more info.

## How it works

This modifies the kernel driver to force enable BAR1 P2P mode on GPUs not intended to use it.
Then, the transfers are done by directly writing to the other GPU physical addresses over DMA.

IOMMU virtualization must be disabled to use the patch, or transfers will fail.
Note that this is very dangerous if you run untrusted software or devices.

Proper IOMMU support is possible but would require registering IOVA mappings.
That would be a lot more work and seems unneeded given the intended use.

## Prerequisites

Enable DMA passthrough mode for IOMMU:

1. Edit `/etc/default/grub`
2. Add `amd_iommu=on iommu=pt` to `GRUB_CMDLINE_LINUX_DEFAULT`
3. Run `sudo update-grub`
4. Reboot

## Installation

### Method 1: Build and install deb package (Recommended)

This method uses DKMS to automatically rebuild the kernel modules when the kernel is updated.

#### Build dependencies

```bash
sudo apt update
sudo apt install build-essential debhelper dh-dkms dkms linux-headers-$(uname -r)
```

#### Build the deb package

```bash
cd /path/to/open-gpu-kernel-modules

# Build the deb package (unsigned)
dpkg-buildpackage -us -uc -b

# The deb file will be in the parent directory
ls ../*.deb
```

#### Install

```bash
# First install the official NVIDIA driver 580 (userspace components)
# Ubuntu: sudo apt install nvidia-driver-580
# Or install from NVIDIA's official .run file

# Remove existing NVIDIA DKMS modules if any
sudo apt remove nvidia-dkms-580 nvidia-dkms-580-open 2>/dev/null || true

# Install the P2P version
sudo dpkg -i ../nvidia-kernel-open-p2p-dkms_580.95.05-1_amd64.deb

# Reboot
sudo reboot
```

### Method 2: Manual installation

1. Install the official NVIDIA driver from https://www.nvidia.com/en-us/drivers/details/254665/
2. Run `./install.sh` in this repo
3. Reboot the server

## Verify installation

```bash
# Check DKMS status (if using deb package)
dkms status

# Check if modules are loaded
lsmod | grep nvidia

# Test GPU
nvidia-smi
```

## Troubleshooting

- If P2P transfers fail, make sure IOMMU is in passthrough mode (`iommu=pt`)
- Check `dmesg | grep -i nvidia` for kernel module errors
- Ensure the kernel headers match your running kernel: `apt install linux-headers-$(uname -r)`
