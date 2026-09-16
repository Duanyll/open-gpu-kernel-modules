#!/usr/bin/env python3
"""Patch x86-64 libcuda for mixed-generation P2P and optional DMA-BUF GDR."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile


# Each rewrite is (byte offset in the signature, replacement bytes).
P2P_PATCHES = (
    (
        "can-access predicate",
        """
        48 8b 83 50 0c 00 00     # mov rax, qword ptr [rbx + 0xc50]
        49 8b 94 24 50 0c 00 00  # mov rdx, qword ptr [r12 + 0xc50]
        48 39 d0                 # cmp rax, rdx
        74 ??                    # je 0x44 -> jmp 0x44
        48 3d c0 00 00 00        # cmp rax, 0xc0
        75 ??                    # jne 0x9
        48 81 fa c8 00 00 00     # cmp rdx, 0xc8
        74 ??                    # je 0x33
        48 3d c8 00 00 00        # cmp rax, 0xc8
        75 ??                    # jne 0x9
        48 81 fa c0 00 00 00     # cmp rdx, 0xc0
        74 ??                    # je 0x22
        83 e0 f0                 # and eax, -0x10
        48 3d f0 00 00 00        # cmp rax, 0xf0
        74 ??                    # je 0xb
        31 c0                    # xor eax, eax
        """,
        (18, b"\xeb"),
    ),
    (
        "compatibility predicate",
        """
        48 8b 87 50 0c 00 00  # mov rax, qword ptr [rdi + 0xc50]
        48 8b 93 50 0c 00 00  # mov rdx, qword ptr [rbx + 0xc50]
        48 39 d0              # cmp rax, rdx
        74 ??                 # je 0x45 -> jmp 0x45
        48 3d c0 00 00 00     # cmp rax, 0xc0
        75 ??                 # jne 0x9
        48 81 fa c8 00 00 00  # cmp rdx, 0xc8
        74 ??                 # je 0x34
        48 3d c8 00 00 00     # cmp rax, 0xc8
        75 ??                 # jne 0x9
        48 81 fa c0 00 00 00  # cmp rdx, 0xc0
        74 ??                 # je 0x23
        83 e0 f0              # and eax, -0x10
        48 3d f0 00 00 00     # cmp rax, 0xf0
        75 ??                 # jne 0xc
        83 e2 f0              # and edx, -0x10
        """,
        (17, b"\xeb"),
    ),
    (
        "peer-device selection",
        """
        49 8b 87 50 0c 00 00  # mov rax, qword ptr [r15 + 0xc50]
        49 8b 96 50 0c 00 00  # mov rdx, qword ptr [r14 + 0xc50]
        48 39 d0              # cmp rax, rdx
        74 ??                 # je 0x7a -> jmp 0x7a
        48 81 fa c8 00 00 00  # cmp rdx, 0xc8
        75 ??                 # jne 0x8
        48 3d c0 00 00 00     # cmp rax, 0xc0
        74 ??                 # je 0x69
        48 81 fa c0 00 00 00  # cmp rdx, 0xc0
        75 ??                 # jne 0x8
        48 3d c8 00 00 00     # cmp rax, 0xc8
        74 ??                 # je 0x58
        83 e0 f0              # and eax, -0x10
        48 3d f0 00 00 00     # cmp rax, 0xf0
        75 ??                 # jne 0xd
        83 e2 f0              # and edx, -0x10
        48 81 fa f0 00 00 00  # cmp rdx, 0xf0
        74 ??                 # je 0x41
        """,
        (17, b"\xeb"),
    ),
    (
        "peer setup",
        """
        48 8b 83 50 0c 00 00     # mov rax, qword ptr [rbx + 0xc50]
        49 8b 94 24 50 0c 00 00  # mov rdx, qword ptr [r12 + 0xc50]
        48 39 d0                 # cmp rax, rdx
        74 ??                    # je 0x53 -> jmp 0x53
        48 3d c0 00 00 00        # cmp rax, 0xc0
        75 ??                    # jne 0x9
        48 81 fa c8 00 00 00     # cmp rdx, 0xc8
        74 ??                    # je 0x42
        48 3d c8 00 00 00        # cmp rax, 0xc8
        75 ??                    # jne 0x9
        48 81 fa c0 00 00 00     # cmp rdx, 0xc0
        74 ??                    # je 0x31
        83 e0 f0                 # and eax, -0x10
        48 3d f0 00 00 00        # cmp rax, 0xf0
        75 ??                    # jne 0x16
        83 e2 f0                 # and edx, -0x10
        48 81 fa f0 00 00 00     # cmp rdx, 0xf0
        74 ??                    # je 0x1a
        """,
        (18, b"\xeb"),
    ),
)


# Harry Chen's device-init capability-bit method, reviewed against 615.71.09:
# https://harrychen.xyz/2026/05/20/enable-gpudirect-rdma-on-rtx-5090/
# https://gist.github.com/Harry-Chen/0d8c941e80c84e5482a46e3f5bcdb63d
# The exact private layout is intentional: layout drift needs disassembly review.
# This adds bit 0x20 without bypassing global GDR or kernel DMA-BUF capability checks.
GDR_PATCHES = (
    (
        "DMA-BUF GDR capability initialization (615 layout)",
        """
        0f b6 87 1c 8f 00 00  # movzx eax, byte ptr [rdi + 0x8f1c]
        89 c2                 # mov edx, eax
        83 ca 40              # or edx, 0x40 -> or edx, 0x60
        83 bf 60 0c 00 00 08  # cmp dword ptr [rdi + 0xc60], 8
        88 97 1c 8f 00 00     # mov byte ptr [rdi + 0x8f1c], dl
        """,
        (11, b"\x60"),
    ),
)


def executable_ranges(data):
    """Limit signatures to file-backed executable ELF64 PT_LOAD segments."""
    if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError("expected a little-endian ELF64 shared library")
    kind, machine, version = struct.unpack_from("<HHI", data, 16)
    if (kind, machine, version) != (3, 62, 1):
        raise ValueError("expected an x86-64 ELF shared library")
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phsize, phnum = struct.unpack_from("<HH", data, 54)
    if phsize != 56 or not phnum or phoff < 64 or phoff + phsize * phnum > len(data):
        raise ValueError("invalid ELF program header table")
    ranges = []
    for index in range(phnum):
        kind, flags, offset, _, _, size, memsize, _ = struct.unpack_from(
            "<IIQQQQQQ", data, phoff + index * phsize
        )
        if kind == 1:
            if size > memsize or offset + size > len(data):
                raise ValueError("invalid ELF load segment")
            if flags & 1 and size:
                ranges.append((offset, offset + size))
    if not ranges:
        raise ValueError("ELF has no executable load segments")
    return ranges


def plan_patches(data, patches):
    ranges = executable_ranges(data)
    changes = []
    for name, signature, (relative, replacement) in patches:
        tokens = re.sub(r"#.*", "", signature).split()
        original = bytes.fromhex(" ".join(tokens[relative:relative + len(replacement)]))
        parts = [b"." if token == "??" else re.escape(bytes.fromhex(token)) for token in tokens]
        parts[relative:relative + len(replacement)] = [
            b"(?:" + re.escape(original) + b"|" + re.escape(replacement) + b")"
        ]
        pattern = re.compile(b"".join(parts), re.DOTALL)
        matches = sorted({match.start() for start, end in ranges
                          for match in pattern.finditer(data, start, end)})
        if len(matches) != 1:
            raise ValueError(f"{name} signature matched {len(matches)} times instead of once")
        offset = matches[0] + relative
        before = data[offset:offset + len(replacement)]
        changes.append((name, offset, before, replacement))
    return changes


def write_library(source, original, patched, output):
    if output is not None:
        # Exclusive creation also refuses existing symlinks and the input itself.
        with output.open("xb") as library:
            library.write(patched)
        shutil.copymode(source, output)
        return

    # Resolve library symlinks, retain a backup of this exact input, then replace
    # the inode atomically so a live process never sees partially patched bytes.
    source = source.resolve(strict=True)
    digest = hashlib.sha256(original).hexdigest()
    backup = source.with_name(source.name + ".bak." + digest[:16])
    try:
        with backup.open("xb") as library:
            library.write(original)
        shutil.copystat(source, backup)
    except FileExistsError:
        if backup.read_bytes() != original:
            raise ValueError(f"backup exists with different contents: {backup}")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=source.parent, prefix=".libcuda-patch-", delete=False) as library:
            temporary = Path(library.name)
            library.write(patched)
            library.flush()
            os.fsync(library.fileno())
        shutil.copystat(source, temporary)
        source_stat = source.stat()
        temp_stat = temporary.stat()
        if (source_stat.st_uid, source_stat.st_gid) != (temp_stat.st_uid, temp_stat.st_gid):
            os.chown(temporary, source_stat.st_uid, source_stat.st_gid)
        if source.read_bytes() != original:
            raise ValueError("source changed during patching; refusing to replace it")
        os.replace(temporary, source)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"backup: {backup}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("libcuda", type=Path, help="input libcuda shared library")
    parser.add_argument("--patch", choices=("p2p", "gdr", "all"), default="p2p",
                        help="patch set (default: p2p; gdr uses the reviewed 615.71.09 layout)")
    parser.add_argument("--dry-run", action="store_true", help="check and show changes without writing")
    parser.add_argument("--output", type=Path, help="write a new copy; refuse to overwrite an existing path")
    args = parser.parse_args(argv)
    patches = {"p2p": P2P_PATCHES, "gdr": GDR_PATCHES, "all": P2P_PATCHES + GDR_PATCHES}[args.patch]

    try:
        original = args.libcuda.read_bytes()
        changes = plan_patches(original, patches)
        data = bytearray(original)
        for _, offset, _, replacement in changes:
            data[offset:offset + len(replacement)] = replacement
        changed = data != original
        if not args.dry_run and (changed or args.output is not None):
            write_library(args.libcuda, original, data, args.output)
        for name, offset, before, after in changes:
            action = "already applied" if before == after else ("would apply" if args.dry_run else "applied")
            print(f"{action}: {name} at 0x{offset:x} ({before.hex()} -> {after.hex()})")
        print(f"input SHA256: {hashlib.sha256(original).hexdigest()}")
        return 0
    except (OSError, ValueError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
