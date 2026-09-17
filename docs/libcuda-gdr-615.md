# DMA-BUF GDR capability patch in libcuda 615.71.09

Updated: 2026-09-17. Scope: x86-64 Linux libcuda 615.71.09, static analysis,
machine-code branch tests, and eight 48 GiB RTX 4090 GPUs on Villa node27.
The complete patch passes capability queries, DMA-BUF export and mlx5 MR registration
on all eight GPUs. End-to-end RDMA and NCCL acceptance are separate checks.

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

The standalone site alone is insufficient in 615: `cuInit` uses two inlined copies.
All three copies must add `0x20` to the initialization OR, preserving `0x40`. Each
also has an alternate assignment from the original register (`OR 0xc0`), which
must become `OR 0xe0` to retain the new capability on that branch. The patch does not
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

## Inlined initializers and runtime verification

The initial standalone-only patch passed static signature checks but failed node27
acceptance: all three attributes remained 0 and export returned 801. Runtime inspection
found the global GDR byte was 1, `dev+0x8e6c` was `0x09`, and `dev+0x8f1c` was `0x58`.
The missing bit was in device initialization, not the global or kernel export checks.

The `cuInit` code at `0x337017` initializes the first device, then a loop at `0x1b3e1e`
initializes the remaining devices. These compiler copies use ECX and R8D respectively,
so Harry's EDX instruction template does not match them. Independent library copies
gave this A/B result without changing the installed library:

| Added patch | Device capability bytes after cuInit |
| --- | --- |
| First-device inline OR only | GPU 0: `0x78`; GPU 1–7: `0x58` |
| Remaining-device inline OR only | GPU 0: `0x58`; GPU 1–7: `0x78` |
| Both inline ORs | GPU 0–7: `0x78` |

The complete GDR rewrite set is:

| Initializer | Normal OR `40 → 60` | Alternate OR `c0 → e0` |
| --- | --- | --- |
| Standalone | `0x4aae5b` | `0x4aae80` |
| First device | `0x337022` | `0x33704f` |
| Remaining devices | `0x1b3e2b` | `0x1b3e51` |

The alternate branches overwrite the capability from the original EAX/ESI value;
changing only the first OR would lose `0x20` there. Unicorn executes both generation
and alternate-store branches for all three reviewed sequences, with several initial
capability values, to check that other bits survive. The node27 Ada cards use the
`generation <= 8` path; the other path has machine-code tests, not Blackwell hardware
validation.

With all six GDR changes, node27 reports attributes 110/116/124 = 1 on every GPU,
exports a 32 MiB DMA-BUF with flags=0, and registers/deregisters it with `mlx5_0`
using `ibv_reg_dmabuf_mr`. This proves export and MR registration, not data transfer.

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
| GDR only | Six sites in the table above | `e08a96ee391219dfb270130e6bb48ee3422033fb75073e734542ec9fc29117ee` |
| P2P + GDR | Six GDR sites plus `0x31ce26`, `0x31cf2b`, `0x31d031`, `0x4b8664` | `db526c80bc77e510c6ef11bc5fc283240971ee657596407cdeefcb2c1bf579ad` |

Byte comparison confirmed exactly six and ten changes respectively, with no
other file changes and an untouched original. Re-running either selection reports
already-applied changes. Thirteen regression tests pass, including a real-binary check,
duplicate signatures, unknown layout, invalid ELF, non-executable decoys, no writes
on incomplete matches, missing inline copies, upgrading the standalone-only patch,
branch execution, independent selections, backups, symlinks, and idempotence:

```sh
LIBCUDA_615=/path/to/original/libcuda.so.615.71.09 uv run --with unicorn python tests/test-libcuda-patch.py -v
```

Without `LIBCUDA_615`, the real-binary test is skipped. Without Unicorn, branch
execution is skipped. Other tests use synthetic ELF fixtures. NVIDIA binaries are
not stored in this repository.

For GPU acceptance, check attributes 110/116/124, an actual
`cuMemGetHandleForAddressRange(... DMA_BUF_FD, flags=0)` export, NIC MR registration,
and verified NCCL transfers in that order. The 590 results do not validate 615, and
the DMA-BUF patch does not unlock legacy `nvidia-peermem`. Method 3/GDR coexistence
on a 48 GB 4090 also needs explicit BAR1 usage and data-integrity testing.
