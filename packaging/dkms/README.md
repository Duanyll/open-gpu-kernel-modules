# nvidia-open-p2p-dkms

A DKMS `.deb` that builds these open GPU kernel modules — with **PCIe peer-to-peer** and a
**dynamic-BAR1 path for 48GB-modded RTX 4090** (see [`docs/48g-4090-p2p`](../../docs/48g-4090-p2p)) —
as a **drop-in replacement** for the CUDA-repo `nvidia-kernel-open-dkms`. The patched source is
vendored into the `.deb`, so DKMS builds it **fully offline** on the target machine.

> Unofficial community fork. Not affiliated with NVIDIA. No warranty — use at your own risk.

## Install (from a release asset)

```sh
sudo apt install ./nvidia-open-p2p-dkms_<ver>-1_all.deb
```

It `Conflicts`/`Replaces` `nvidia-kernel-open-dkms` and `Provides: nvidia-kernel-<ver>`, so apt
swaps the stock kernel package cleanly and the userspace (`nvidia-driver-cuda`, `libcuda1`, …) stays
in place. The postinst builds and installs the modules for every installed kernel that has headers;
it does **not** reload the running driver — the new modules take effect on the next reboot.

**Supported base:** Debian / NVIDIA CUDA apt repo package names (`nvidia-kernel-open-dkms`,
`firmware-nvidia-gsp = <ver>`). Match the `<ver>` to the branch you build from. Other distros
(e.g. Ubuntu's `nvidia-dkms-<ver>-open`) use different names and are not targeted.

## Build

```sh
packaging/dkms/build-deb.sh            # -> packaging/dkms/nvidia-open-p2p-dkms_<ver>-1_all.deb
```

Needs `dpkg-deb`. Version is read from `version.mk` (`NVIDIA_VERSION`), so the same packaging works
on every `<ver>-p2p[-48g]` branch — nothing here is version-specific.

## Why only `dkms.conf` is hand-written

The tree ships `kernel-open/dkms.conf` as an `nvidia-installer` **template** (`__VERSION_STRING`,
`__DKMS_MODULES`, …) that DKMS can't consume. This package drops it and generates a **processed**
root `dkms.conf` (explicit 5-module list). Every other build-glue file (`Makefile`, `Kbuild`,
`conftest.sh`, …) is identical to the stock `nvidia-kernel-open-dkms`, so nothing else needs porting.

## Files
| file | role |
| --- | --- |
| `build-deb.sh` | reads `version.mk`, vendors the source keep-list, generates the files below, assembles the `.deb`. |
| `dkms.conf.in` | processed dkms.conf template (5 modules → `/updates/dkms`). |
| `control.in` | `Provides: nvidia-kernel-<ver>` + `Conflicts/Replaces: nvidia-kernel-open-dkms`. |
| `postinst.in` | offline `dkms build/install` for all kernels with headers. |
| `prerm.in` | `dkms remove`. |
