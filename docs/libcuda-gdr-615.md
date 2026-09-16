# DMA-BUF GDR capability patch in libcuda 615.71.09

Date: 2026-09-16. Scope: x86-64 Linux libcuda, static binary analysis and patcher
tests. No GPU, DMA-BUF export, NIC memory registration, or NCCL runtime validation
has been performed with this 615 patch.

## Finding

The initialization site described in [Harry Chen's investigation](https://harrychen.xyz/2026/05/20/enable-gpudirect-rdma-on-rtx-5090/)
is still present in 615.71.09. Both its file location and the private capability
field have moved. This was checked against downloaded NVIDIA binaries, not
inferred solely from a pattern match.

| Item | 590.48.01 | 615.71.09 |
| --- | --- | --- |
| Capability byte in private device object | `+0x5f14` | `+0x8f1c` |
| GDR capability mask | `0x20` | `0x20` |
| Initialization sequence begins | `0x42d690` | `0x4aae50` |
| Patch byte, file offset | `0x42d69b` | **`0x4aae5b`** |
| Original → patched instruction | `83 ca 40` → `83 ca 60` | `83 ca 40` → `83 ca 60` |
| Compared private field / immediate | `+0xc60` / `8` | `+0xc60` / `8` |

The patch adds `0x20` to the initialization OR, preserving `0x40`. It does not
replace an API's result with success or bypass the global GDR and DMA-BUF checks.
The author's [automatic locator](https://gist.github.com/Harry-Chen/0d8c941e80c84e5482a46e3f5bcdb63d/63b6be7e7349f9ae95f100299fd46d974175987f)
recognizes the instruction shape but defaults to the old private layout, so 615
requires manual review rather than blindly accepting layout drift.

## Binary provenance

Downloaded from NVIDIA's Debian 13 x86-64 repository. Package SHA-256 values were
checked against its `Packages.gz` index before extraction. No package was installed.

| Package | SHA-256 of `.deb` |
| --- | --- |
| [590.48.01-1](https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/libcuda1_590.48.01-1_amd64.deb) | `408435f219b6d4c57fb091e1b2e1e5b35f4296601abeceb60d6d4573758c13fc` |
| [615.71.09-1](https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/libcuda1_615.71.09-1_amd64.deb) | `44d19ef4aa036b85d6d93c0245f52a7bc1ede1561c6cf0e69ea083593c9d6793` |
| [615.71.09-2](https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/libcuda1_615.71.09-2_amd64.deb) | `75643aab978f99c52d6e8e40ac28dd56182ca187d2dc1de581e755ac2ecd10c6` |

The two 615 packages contain identical `libcuda.so.615.71.09`: 112,475,080 bytes,
SHA-256 `85a8cc8cf00419ceb295ec62c481d05b4cdbdab10f25c5375d6d24bdbe00b4d9`.
The extracted 590 library has SHA-256
`e24ade728a7b2280b60f5603855db23ca137d87d342ecac4219157d2cc5c3694` and reproduces
the original blog's initialization address and bytes.

## Attribute-to-initializer trace in 615

The exported `cuDeviceGetAttribute` wrapper at ELF VA `0x373c90` calls through
slot `0x6b3fcc8`; its ELF relocation resolves to implementation `0x372790`.
That implementation indexes the device table and calls attribute dispatcher
`0x2afce0`. The dispatcher uses signed 32-bit relative entries at `0x6440918`.

| CUDA attribute | ID | Branch VA | Checks in this binary |
| --- | --- | --- | --- |
| `GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED` | 110 | `0x2b0566` | VMM helper, `dev+0x8f1c & 0x20`, global GDR byte |
| `GPU_DIRECT_RDMA_SUPPORTED` | 116 | `0x2b0b89` | `dev+0x8f1c & 0x20`, global GDR byte |
| `DMA_BUF_SUPPORTED` | 124 | `0x2affd9` | Additional device check, `dev+0x8e6c & 0x08`, `dev+0x8f1c & 0x20`, global GDR byte |

All three branches use the global byte at `0x6b53a98`. Attribute 124 also has a
conditional path via `0x2b0ff5` involving `dev+0x8e61` and `dev+0xc60`; that
condition is left intact. Attribute IDs come from the
[CUDA Driver API enum](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html);
the private fields and addresses above come from local Capstone/ELF analysis.

References to the capability field lead to this executable sequence:

```text
0x4aae50  0f b6 87 1c 8f 00 00     movzx eax, byte ptr [rdi + 0x8f1c]
0x4aae57  89 c2                    mov   edx, eax
0x4aae59  83 ca 40                 or    edx, 0x40
0x4aae5c  83 bf 60 0c 00 00 08     cmp   dword ptr [rdi + 0xc60], 8
0x4aae63  88 97 1c 8f 00 00        mov   byte ptr [rdi + 0x8f1c], dl
```

These instruction VAs equal their file offsets in the executable segment of this
binary. That equality is not assumed by the patcher: it reports file offsets and
searches only file-backed executable `PT_LOAD` ranges. The reviewed 615 GDR pattern
fixes both private-field displacements and the comparison immediate. Future
layout changes are rejected instead of silently accepted.

This establishes the same initializer/attribute relationship as the 590 method.
It does not establish all initialization paths or the complete DMA-BUF export
call chain on a running 615 driver. In particular, attribute success alone would
not prove successful export or GDR transport.

## Integration and verification

`patch-libcuda-p2p.py` retains P2P-only behavior by default. `--patch gdr` selects
the reviewed capability change; `--patch all` selects it and the original four
P2P changes. The GDR locator is implemented in this script's signature engine;
it does not execute or vendor the author's Gist.

All selected matches are validated before writing, including when some changes
are already applied. An explicit output path is created exclusively. In-place
operation preserves a hash-named backup and replaces the resolved library file
atomically. The [README](../README.md#optional-libcuda-patches) gives usage examples.

Actual 615 binary checks passed:

| Selection | Changed file offsets | Output SHA-256 |
| --- | --- | --- |
| GDR only | `0x4aae5b` | `81e631b7ccb8eceb6771ee3c5e7d645adac6e11b0d970351c4c967ed4df0b8da` |
| P2P + GDR | `0x31ce26`, `0x31cf2b`, `0x31d031`, `0x4aae5b`, `0x4b8664` | `d1c67b4611680afc093d8d364edd82ca7e56c98fed02c3b8efaa2d96f664eb4a` |

Byte comparison confirmed exactly one and five changes respectively, with no
other file changes and an untouched original. Re-running either selection reports
already-applied changes. Ten regression tests pass, including a real-binary check,
duplicate signatures, unknown layout, invalid ELF, non-executable decoys, no writes
on incomplete matches, independent selections, backups, symlinks, and idempotence:

```sh
LIBCUDA_615=/path/to/original/libcuda.so.615.71.09 python3 tests/test-libcuda-patch.py -v
```

Without `LIBCUDA_615`, the real-binary test is skipped; the remaining tests use
synthetic ELF fixtures. NVIDIA binaries are not stored in this repository.

For GPU acceptance, check attributes 110/116/124, an actual
`cuMemGetHandleForAddressRange(... DMA_BUF_FD, flags=0)` export, NIC MR registration,
and verified NCCL transfers in that order. The 590 results do not validate 615, and
the DMA-BUF patch does not unlock legacy `nvidia-peermem`. Method 3/GDR coexistence
on a 48 GB 4090 also needs explicit BAR1 usage and data-integrity testing.
