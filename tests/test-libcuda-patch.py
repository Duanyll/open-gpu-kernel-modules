#!/usr/bin/env python3
"""Patcher regression tests; LIBCUDA_615 optionally selects an official binary."""

import contextlib
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import re
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("patcher", ROOT / "patch-libcuda-p2p.py")
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)

# Reviewed 615 initialization sequence, including the matching load/store field.
GDR = bytes.fromhex("0f b6 87 1c 8f 00 00 89 c2 83 ca 40 83 bf 60 0c 00 00 08 88 97 1c 8f 00 00")


def elf(code, trailer=b""):
    image = bytearray(0x100)
    image[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", image, 16, 3, 62, 1)
    struct.pack_into("<Q", image, 32, 64)
    struct.pack_into("<HHH", image, 52, 64, 56, 1)
    struct.pack_into("<IIQQQQQQ", image, 64, 1, 5, 0x100, 0x100, 0,
                     len(code), len(code), 0x1000)
    return bytes(image) + code + trailer


def p2p_code():
    # Synthetic jump displacements exercise scanning, not x86 execution.
    return b"\x90".join(bytes.fromhex(re.sub(r"#.*", "", signature).replace("??", "0a"))
                        for _, signature, _ in patcher.P2P_PATCHES)


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "libcuda.so.615.71.09"
        self.original = elf(p2p_code() + b"\x90" + GDR)
        self.source.write_bytes(self.original)
        self.source.chmod(0o644)

    def run_cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return patcher.main([str(self.source), *map(str, args)])

    def test_gdr_location_and_nonexecutable_decoy(self):
        plan = patcher.plan_patches(elf(GDR, GDR), patcher.GDR_PATCHES)
        self.assertEqual(plan[0][1:], (0x10b, b"\x40", b"\x60"))

    def test_ambiguous_or_changed_layout_rejected(self):
        variants = [GDR + GDR, GDR.replace(b"\x1c\x8f", b"\x14\x5f"),
                    GDR[:21] + b"\x20\x8f\x00\x00",
                    GDR[:11] + b"\x80" + GDR[12:],
                    GDR[:18] + b"\x09" + GDR[19:]]
        for code in variants:
            with self.subTest(code=code), self.assertRaises(ValueError):
                patcher.plan_patches(elf(code), patcher.GDR_PATCHES)

    def test_invalid_elf_rejected(self):
        for position, replacement in ((0, b"BAD!"), (4, b"\x01"), (5, b"\x02"),
                                      (16, b"\x02\x00"), (18, b"\xb7\x00"),
                                      (54, b"\xff\xff"), (96, b"\xff" * 8)):
            image = bytearray(self.original)
            image[position:position + len(replacement)] = replacement
            with self.subTest(position=position), self.assertRaises(ValueError):
                patcher.plan_patches(image, patcher.GDR_PATCHES)
        with self.assertRaises(ValueError):
            patcher.plan_patches(b"\x7fELF", patcher.GDR_PATCHES)

    def test_dry_run_creates_nothing(self):
        output = self.source.with_name("copy")
        self.assertEqual(self.run_cli("--patch", "all", "--dry-run", "--output", output), 0)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(list(self.source.parent.iterdir()), [self.source])

    def test_last_signature_failure_prevents_all_writes(self):
        self.source.write_bytes(elf(p2p_code()))
        before = self.source.read_bytes()
        self.assertEqual(self.run_cli("--patch", "all"), 1)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(list(self.source.parent.iterdir()), [self.source])

    def test_combined_copy_changes_only_five_bytes(self):
        output = self.source.with_name("combined")
        self.assertEqual(self.run_cli("--patch", "all", "--output", output), 0)
        after = output.read_bytes()
        self.assertEqual(sum(a != b for a, b in zip(after, self.original)), 5)
        self.assertEqual(len(after), len(self.original))
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(output.stat().st_mode & 0o777, 0o644)
        self.assertTrue(all(before == replacement for _, _, before, replacement
                            in patcher.plan_patches(after, patcher.P2P_PATCHES + patcher.GDR_PATCHES)))

    def test_existing_output_is_untouched(self):
        output = self.source.with_name("existing")
        output.write_bytes(b"keep this file")
        self.assertEqual(self.run_cli("--patch", "gdr", "--output", output), 1)
        self.assertEqual(output.read_bytes(), b"keep this file")
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_independent_sets_backup_and_idempotence(self):
        self.assertEqual(self.run_cli(), 0)  # Existing invocation remains P2P-only.
        p2p = self.source.read_bytes()
        self.assertEqual(sum(a != b for a, b in zip(p2p, self.original)), 4)
        self.assertEqual(patcher.plan_patches(p2p, patcher.GDR_PATCHES)[0][2], b"\x40")
        self.assertEqual(self.run_cli("--patch", "gdr"), 0)
        combined = self.source.read_bytes()
        self.assertEqual(sum(a != b for a, b in zip(p2p, combined)), 1)
        backups = sorted(self.source.parent.glob("*.bak.*"))
        self.assertEqual(len(backups), 2)
        self.assertEqual({p.read_bytes() for p in backups}, {self.original, p2p})
        self.assertEqual(self.run_cli("--patch", "all"), 0)
        self.assertEqual(self.source.read_bytes(), combined)
        self.assertEqual(sorted(self.source.parent.glob("*.bak.*")), backups)

    def test_symlink_keeps_target_and_original_backup(self):
        real = self.source
        self.source = real.with_name("libcuda.so.1")
        self.source.symlink_to(real.name)
        self.assertEqual(self.run_cli("--patch", "gdr"), 0)
        self.assertTrue(self.source.is_symlink())
        self.assertEqual(len(list(real.parent.glob(real.name + ".bak.*"))), 1)
        self.assertEqual(sum(a != b for a, b in zip(real.read_bytes(), self.original)), 1)

    @unittest.skipUnless(os.environ.get("LIBCUDA_615"), "set LIBCUDA_615 for real-binary validation")
    def test_official_615_binary(self):
        data = Path(os.environ["LIBCUDA_615"]).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         "85a8cc8cf00419ceb295ec62c481d05b4cdbdab10f25c5375d6d24bdbe00b4d9")
        plan = patcher.plan_patches(data, patcher.P2P_PATCHES + patcher.GDR_PATCHES)
        self.assertEqual([offset for _, offset, _, _ in plan],
                         [0x31ce26, 0x31d031, 0x4b8664, 0x31cf2b, 0x4aae5b])
        self.assertEqual([before for _, _, before, _ in plan], [b"\x74"] * 4 + [b"\x40"])


if __name__ == "__main__":
    unittest.main()
