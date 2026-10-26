# Method 3 port to 615.71.09

Date: 2026-09-16. Branch: `615.71.09-p2p-48g`.

This port runs in production on 4- and 8-GPU 48 GB RTX 4090 nodes, with NCCL
`P2P/CUMEM` inside each node and DMA-BUF GPUDirect RDMA over ConnectX-5 100G RoCE
between nodes. The other design notes and hardware logs in this directory come from
the earlier 590/595 work.

## Sources

| Layer | Source |
| --- | --- |
| NVIDIA 615.71.09 | `61dcc93722ecb418bb5f2e00923f05b4b8051dd1` |
| aikitoria `615.71.09-p2p` | `c8bff5400e2fc8bda4cb9a141fb596aad607a35f` |
| Local Method 3 source | `f6eb773b17d4c976ab1a4a91c2ed8ee1e2d8cd79`, diff from aikitoria 590 base `b4707703552298557a38e9a046403e7ad501109d` |

The complete aikitoria 615 stack is retained: PCIe/NVLink selection, cross-generation
P2P and its `libcuda` patch, static BAR1 firmware-console sharing, and experimental
hugetlb `cudaHostRegister`. Cross-architecture PCIe atomics remain disabled.

The userspace patcher also has an independently selectable DMA-BUF/GDR capability
patch for the reviewed 615 libcuda layout. Its binary analysis and test scope are
recorded in [libcuda-gdr-615.md](../libcuda-gdr-615.md); it is not applied by the DKMS package.

NVIDIA 615 already releases `session->devicesLock` before `pRmApi->Free` in
`nvGpuOpsRmDeviceDestroy`, fixing the teardown side of the observed session/client
lock inversion. This branch also removes the mapping-side lock acquisition:
`_nvGpuOpsIsBar1P2pAtomicEnabled` reads architecture from the live `OBJGPU` objects
instead of looking up the session device tree. No session lock is needed for that
immutable metadata.

## Method 3 integration

The three driver files changed are `kern_bus_gh100.c`, `p2p_api.c`, and `nv_gpu_ops.c`.
The dynamic path is still scoped to explicit external allocations on homogeneous
48 GB AD102 nodes:

- Each duped allocation owns a contiguous BAR1 VA window on the remote GPU. A
  source-side IOMMU mapping supplies the DMA base, and the duped-handle release
  tears down that window. There is also a subdevice teardown cleanup path.
- The window descriptor belongs to the remote GPU so the OS IOVA mapper selects
  the peer BAR mapping path. GPU framebuffer pages themselves need not be contiguous.
- Static pairs retain the 615 whole-framebuffer IOMMU mappings. Dynamic pairs
  skip those mappings; mixed static/dynamic pairs are rejected.
- Dynamic capability queries report no global DMA window (`0/0`). Managed-memory
  peer migration and UVM physical peer copies remain outside the supported scope;
  this port does not add a managed-memory gate.
- The 615 GMMU/PTE interfaces and static BAR1 address checks are preserved. A
  separate dynamic encoder converts allocation-relative offsets through those
  checks, rather than treating physical framebuffer offsets as window offsets.

Port-specific hardening:

- An explicit boolean selects dynamic mapping, so IOVA zero is valid.
- The effective mapping size is checked against `memdescGetSize`, including when
  `size == 0` expands to an allocation's padded `ActualSize`.
- Range and budget checks use subtraction to avoid unsigned addition overflow;
  BAR1 physical-base construction uses `portSafeAddU64`.
- The 64 MiB BAR1 reserve remains a source-subdevice heuristic. It is not a global
  reservation across peers or NIC mappings; the aperture allocator is authoritative.

The existing Method 3 mapping cache and teardown model are retained.

## Validation

- `python3 tests/test-method3-host.py` compiles the production address encoders and
  atomic policy with small host stubs and UBSan. It covers static range rejection,
  dynamic high-FB offsets, partial output, IOVA zero, end-of-window boundaries,
  unsigned overflow, and same-/cross-architecture atomic policy. It passes with
  Clang on macOS ARM64 and GCC 14 on Linux ARM64.
- All five modules build for x86_64 against Debian
  `6.12.107+deb13-amd64`, using GCC 14 in an ARM64 Debian Trixie container with the
  native x86_64 cross toolchain.
- The build is not warning-free: objtool reports 11,668 `naked return found in
  MITIGATION_RETHUNK build` warnings for `nvidia.o`. NVIDIA's unchanged 615 RM
  Makefile lacks `-mfunction-return=thunk-extern`, while its modeset Makefile includes
  that option. No warnings are suppressed by this port.
- `modinfo` reports `615.71.09` and the expected `6.12.107+deb13-amd64` vermagic
  for `nvidia`, `nvidia-modeset`, `nvidia-drm`, `nvidia-uvm`, and `nvidia-peermem`.
- `nvidia-open-p2p-dkms_615.71.09-1_all.deb` builds successfully. Its build inputs
  match the committed source byte for byte, with only the installer DKMS template
  replaced by the processed five-module configuration.

Reproduction after installing the cross compiler, common headers, native kbuild
tools, and extracting the matching amd64 headers:

```sh
make -j8 TARGET_ARCH=x86_64 ARCH=x86_64 \
  CC=x86_64-linux-gnu-gcc CXX=x86_64-linux-gnu-g++ \
  LD=x86_64-linux-gnu-ld OBJCOPY=x86_64-linux-gnu-objcopy \
  OBJDUMP=x86_64-linux-gnu-objdump HOST_CC=gcc \
  KERNEL_UNAME=6.12.107+deb13-amd64 \
  SYSSRC=/usr/src/linux-headers-6.12.107+deb13-common \
  SYSOUT=/usr/src/linux-headers-6.12.107+deb13-amd64 modules
```

Use `SYSSRC` and `SYSOUT` for Debian's split header layout; setting just
`KERNEL_SOURCES`/`KERNEL_OUTPUT` does not pass the required `KBUILD_OUTPUT` to make.

The offline Debian DKMS packaging is carried forward and reads `615.71.09` from
`version.mk`. It packages committed source and must be used with matching NVIDIA
userspace and GSP firmware. See [the packaging README](../../packaging/dkms/README.md).

## Hardware validation

Validated on homogeneous 48 GB RTX 4090 nodes with 615.71.09 userspace/GSP firmware
and `iommu=pt`: NCCL runs `via P2P/CUMEM` with correct collectives while resident
memory exceeds 32 GiB, and cross-node all-reduce runs over DMA-BUF GDR with the
patched `libcuda` ([libcuda-gdr-615.md](../libcuda-gdr-615.md)).

Legacy whole-device peer access still requires all peer-visible allocations to fit
in BAR1. Managed-memory peer migration does not use Method 3.
