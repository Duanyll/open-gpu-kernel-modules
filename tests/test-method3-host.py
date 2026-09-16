#!/usr/bin/env python3
"""Compile the production BAR1 encoders and atomic policy without a GPU.

Only GPU metadata and nvport/logging dependencies are stubbed. This does not
exercise BAR1 allocation, IOMMU mapping, RM locking, or hardware teardown.
Run from any directory; CC selects the host C compiler (default: cc).
"""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/nvidia/src/kernel/rmapi/nv_gpu_ops.c"


def function(source, name):
    match = re.search(
        rf"(?m)^(?:static\s+)?(?:NV_STATUS|NvBool)\s+{re.escape(name)}\s*\(",
        source,
    )
    if match is None:
        raise RuntimeError(f"Cannot find production function {name}")
    start = source.index("{", match.end())
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


PREAMBLE = r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "nvtypes.h"
#include "nvstatus.h"
#include "nvmisc.h"
#include "nv_arch.h"

#define NV_PRINTF(...) ((void)0)
typedef struct { NvU32 arch; } OBJGPU;
struct gpuSession;
static NvU32 gpuGetChipArch(OBJGPU *gpu) { return gpu->arch; }
static NvBool portSafeAddU64(NvU64 a, NvU64 b, NvU64 *result)
{
    return !__builtin_add_overflow(a, b, result);
}
"""

CASES = r"""
int main(void)
{
    const NvU64 page = 64ULL << 10;
    const NvU64 base = 128ULL << 30;
    NvU64 addresses[3] = {0, page, 2 * page};
    OBJGPU ada = {GPU_ARCHITECTURE_ADA};
    OBJGPU gb1 = {GPU_ARCHITECTURE_BLACKWELL_GB1XX};
    OBJGPU gb2 = {GPU_ARCHITECTURE_BLACKWELL_GB2XX};

    // Static mappings preserve framebuffer offsets and enforce the full page.
    assert(_nvGpuOpsEncodeBar1P2PAddrs(addresses, base, 3 * page, page, 3) == NV_OK);
    assert(addresses[0] == base && addresses[2] == base + 2 * page);
    addresses[0] = 3 * page;
    assert(_nvGpuOpsEncodeBar1P2PAddrs(addresses, base, 3 * page, page, 1)
           == NV_ERR_INVALID_ADDRESS);
    addresses[0] = 2 * page + 1;
    assert(_nvGpuOpsEncodeBar1P2PAddrs(addresses, base, 3 * page, page, 1)
           == NV_ERR_INVALID_ADDRESS);
    addresses[0] = page;
    assert(_nvGpuOpsEncodeBar1P2PAddrs(addresses, UINT64_MAX - page + 1,
                                    3 * page, page, 1) == NV_ERR_INVALID_ADDRESS);
    addresses[0] = 0;
    assert(_nvGpuOpsEncodeBar1P2PAddrs(addresses, 0, page, page, 1) == NV_OK);

    // Physical FB pages above 32 GiB become offsets into a small BAR1 window.
    // A short output buffer may encode only the first part of the request.
    addresses[0] = 40ULL << 30;
    addresses[1] = 44ULL << 30;
    addresses[2] = UINT64_MAX;
    assert(_nvGpuOpsEncodeDynBar1P2PAddrs(addresses, base, 4 * page,
                                        page, 3 * page, page, 2) == NV_OK);
    assert(addresses[0] == base + page && addresses[1] == base + 2 * page);
    assert(addresses[2] == UINT64_MAX);

    // Zero is a valid IOVA; the final page of a window is valid too.
    assert(_nvGpuOpsEncodeDynBar1P2PAddrs(addresses, 0, 4 * page,
                                        0, page, page, 1) == NV_OK);
    assert(addresses[0] == 0);
    assert(_nvGpuOpsEncodeDynBar1P2PAddrs(addresses, base, 4 * page,
                                        3 * page, page, page, 1) == NV_OK);
    assert(addresses[0] == base + 3 * page);

    // Reject an out-of-window request even when the output covers fewer pages.
    assert(_nvGpuOpsEncodeDynBar1P2PAddrs(addresses, base, 4 * page,
                                        3 * page, 2 * page, page, 1)
           == NV_ERR_INVALID_LIMIT);
    assert(_nvGpuOpsEncodeDynBar1P2PAddrs(addresses, base, 4 * page,
                                        4 * page, 0, page, 0) == NV_ERR_INVALID_LIMIT);
    assert(_nvGpuOpsEncodeDynBar1P2PAddrs(addresses, base, 4 * page,
                                        page, UINT64_MAX, page, 1) == NV_ERR_INVALID_LIMIT);
    assert(_nvGpuOpsEncodeDynBar1P2PAddrs(addresses, base, UINT64_MAX,
                                        UINT64_MAX, 1, 1, 1) == NV_ERR_INVALID_LIMIT);
    assert(_nvGpuOpsEncodeDynBar1P2PAddrs(addresses, base, 4 * page,
                                        0, page, page, UINT64_MAX)
           == NV_ERR_INVALID_ARGUMENT);
    assert(_nvGpuOpsEncodeDynBar1P2PAddrs(addresses, base, 4 * page,
                                        0, page, 0, 1) == NV_ERR_INVALID_ARGUMENT);
    assert(_nvGpuOpsEncodeDynBar1P2PAddrs(addresses, base, 4 * page,
                                        0, 0, page, 1) == NV_ERR_INVALID_ARGUMENT);
    assert(_nvGpuOpsEncodeDynBar1P2PAddrs(addresses, UINT64_MAX - page + 1,
                                        4 * page, page, page, page, 1)
           == NV_ERR_INVALID_ADDRESS);

    // No session or lock callbacks are available: architecture must come from
    // the live GPU objects. Cross-architecture atomics must remain disabled.
    assert(!_nvGpuOpsIsBar1P2pAtomicEnabled(NULL, &ada, &ada));
    assert(!_nvGpuOpsIsBar1P2pAtomicEnabled(NULL, &gb2, &ada));
    assert(!_nvGpuOpsIsBar1P2pAtomicEnabled(NULL, &ada, &gb2));
    assert(!_nvGpuOpsIsBar1P2pAtomicEnabled(NULL, &gb1, &gb2));
    assert(_nvGpuOpsIsBar1P2pAtomicEnabled(NULL, &gb1, &gb1));
    assert(_nvGpuOpsIsBar1P2pAtomicEnabled(NULL, &gb2, &gb2));
    assert(_nvGpuOpsIsBar1P2pAtomicEnabled(NULL, NULL, &gb2));
    puts("PASS: static/dynamic BAR1 address bounds and P2P atomic policy");
    return 0;
}
"""


def main():
    source = SOURCE.read_text()
    production = "\n\n".join(function(source, name) for name in (
        "_nvGpuOpsEncodeBar1P2PAddrs",
        "_nvGpuOpsEncodeDynBar1P2PAddrs",
        "_nvGpuOpsIsBar1P2pAtomicEnabled",
    ))
    with tempfile.TemporaryDirectory(prefix="method3-host-") as directory:
        test_source = Path(directory) / "test.c"
        binary = Path(directory) / "test"
        test_source.write_text(PREAMBLE + production + CASES)
        subprocess.run([
            *shlex.split(os.environ.get("CC", "cc")),
            "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
            "-fsanitize=undefined", "-fno-sanitize-recover=all",
            "-I", str(ROOT / "src/common/sdk/nvidia/inc"),
            "-I", str(ROOT / "src/common/inc/swref/published"),
            str(test_source), "-o", str(binary),
        ], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
