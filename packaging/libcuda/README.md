# libcuda1-villa

`libcuda1-villa` replaces the NVIDIA CUDA driver library with the reviewed P2P and
DMA-BUF GDR patches. It installs the library at the standard host path, retains the
original SONAME and links, and updates the linker cache through the ldconfig trigger.
It targets Debian 13 amd64 and NVIDIA's CUDA repository package layout.

## Build

Download the official deb named in [`release.json`](../release.json), then run on Debian
with Python 3.11 or newer and `dpkg-deb`:

```sh
python3 packaging/libcuda/build-deb.py \
  libcuda1_615.71.09-2_amd64.deb \
  libcuda1-villa_615.71.09-2+villa2_amd64.deb
```

The build checks the official package and library SHA-256 values, applies all ten
reviewed byte changes, and verifies the final library hash. Unknown inputs and existing
output files are rejected. Building and installing the package require no network access
once the input deb and installation dependencies are available.

## Install and verify

On the test host, select the matching NVIDIA userspace and custom kernel package together:

```sh
sudo apt-get --simulate install \
  nvidia-driver-cuda=615.71.09-2 \
  nvidia-open-p2p-dkms=615.71.09-2+villa2 \
  libcuda1-villa=615.71.09-2+villa2
```

Review the proposed dependency changes and any existing driver pinning, then repeat
without `--simulate`. Reboot into the matching kernel modules before starting GPU tests.
The package revision is separate from the provided upstream version, so NVIDIA packages
requiring `libcuda1 (= 615.71.09-2)` can use this replacement.

```sh
dpkg -V libcuda1-villa
sha256sum /usr/lib/x86_64-linux-gnu/libcuda.so.615.71.09
```

The expected library SHA-256 is
`db526c80bc77e510c6ef11bc5fc283240971ee657596407cdeefcb2c1bf579ad`.
The input hash and individual changes are installed in
`/usr/share/doc/libcuda1-villa/build-manifest.json`.

For Enroot acceptance, check the library loaded by the actual CUDA process and its hash
inside the container. A CUDA compatibility library or explicit library path can override
the host library. Hardware acceptance includes peer access, NCCL and DMA-BUF RDMA;
the binary checks alone do not establish GPU compatibility.

## Restore the official library

```sh
sudo apt-get --simulate install libcuda1=615.71.09-2 libcuda1-villa-
```

Review and repeat without `--simulate`. This restores the official library for the same
driver release; removing `libcuda1-villa` alone does not restore it. Start new jobs and
containers to use the replacement, since existing processes and bind mounts may retain
the previous library. Rolling the whole driver back also requires matching kernel
modules, firmware and userspace.
