# nvidia-open-p2p-dkms

A DKMS `.deb` that builds these open GPU kernel modules — with **PCIe peer-to-peer** and a
**dynamic-BAR1 path for 48GB-modded RTX 4090** (see [`docs/48g-4090-p2p`](../../docs/48g-4090-p2p)) —
as a **drop-in replacement** for the CUDA-repo `nvidia-kernel-open-dkms`. The patched source is
vendored into the `.deb`, so DKMS builds it **fully offline** on the target machine.

> Unofficial community fork. Not affiliated with NVIDIA. No warranty — use at your own risk.

## Install (from a release asset)

```sh
sudo apt install ./nvidia-open-p2p-dkms_615.71.09-2+villa1_all.deb
```

It `Conflicts`/`Replaces` `nvidia-kernel-open-dkms` and `Provides: nvidia-kernel-<ver>`, so apt
swaps the stock kernel package. Its exact firmware dependency requires the corresponding
NVIDIA package release. The postinst builds and installs the modules for every installed kernel that has headers;
it does **not** reload the running driver — the new modules take effect on the next reboot.

**Supported base:** Debian / NVIDIA CUDA apt repo package names (`nvidia-kernel-open-dkms`,
`firmware-nvidia-gsp`). Install the NVIDIA userspace libraries and GSP firmware matching
the kernel module version (`615.71.09` on this branch) before rebooting. Other distros
(e.g. Ubuntu's `nvidia-dkms-<ver>-open`) use different names and are not targeted.

## Build

```sh
packaging/dkms/build-deb.sh
```

Needs `dpkg-deb` and Python 3. Package and upstream dependency versions come from
[`release.json`](../release.json); the driver version must match `version.mk`.
The package filename includes its Villa revision, currently `615.71.09-2+villa1`.

In a Git checkout, the package contains source from **committed `HEAD`**. Commit source
changes before building; uncommitted edits are not included. The 615 port's current
validation scope is recorded in [615-port.md](../../docs/48g-4090-p2p/615-port.md).

The separately packaged [patched libcuda](../libcuda/README.md) provides the P2P and
DMA-BUF GDR userspace changes. Install the two packages with matching NVIDIA userspace
before testing the driver on a GPU host.

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
