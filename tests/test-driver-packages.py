#!/usr/bin/env python3
"""Validate real release debs, including payload ownership and binary changes."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE = json.loads((ROOT / "packaging/release.json").read_text())


def field(deb, name):
    return subprocess.check_output(["dpkg-deb", "-f", str(deb), name], text=True).strip()


def check_archive(deb):
    data = subprocess.check_output(["dpkg-deb", "--fsys-tarfile", str(deb)])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = archive.getmembers()
        assert members[0].name == "." and members[0].mode == 0o755
        assert all(m.uid == 0 and m.gid == 0 for m in members)
        assert all(m.mode & 0o022 == 0 for m in members if not m.issym())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dkms_deb", type=Path)
    parser.add_argument("libcuda_deb", type=Path)
    parser.add_argument("official_deb", type=Path)
    args = parser.parse_args()
    upstream = RELEASE["upstream_version"]
    driver = RELEASE["driver_version"]
    for deb, name, arch in ((args.dkms_deb, "nvidia-open-p2p-dkms", "all"),
                            (args.libcuda_deb, "libcuda1-villa", "amd64")):
        assert field(deb, "Package") == name
        assert field(deb, "Version") == RELEASE["package_version"]
        assert field(deb, "Architecture") == arch
        check_archive(deb)
    assert f"nvidia-kernel-open-dkms (= {upstream})" in field(args.dkms_deb, "Provides")
    assert f"firmware-nvidia-gsp (= {upstream})" in field(args.dkms_deb, "Depends")
    assert f"libcuda1 (= {upstream})" in field(args.libcuda_deb, "Provides")
    assert field(args.libcuda_deb, "Conflicts") == "libcuda1"
    assert field(args.libcuda_deb, "Replaces") == "libcuda1"
    assert field(args.libcuda_deb, "Depends") == field(args.official_deb, "Depends")
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        custom, official = work / "custom", work / "official"
        for deb, target in ((args.libcuda_deb, custom), (args.official_deb, official)):
            subprocess.run(["dpkg-deb", "--raw-extract", str(deb), str(target)], check=True)
        relative = Path(f"usr/lib/x86_64-linux-gnu/libcuda.so.{driver}")
        before, after = (official / relative).read_bytes(), (custom / relative).read_bytes()
        assert hashlib.sha256(before).hexdigest() == RELEASE["libcuda_original_sha256"]
        assert hashlib.sha256(after).hexdigest() == RELEASE["libcuda_patched_sha256"]
        assert len(before) == len(after)
        changes = [(i, a, b) for i, (a, b) in enumerate(zip(before, after)) if a != b]
        assert changes == [(0x31ce26, 0x74, 0xeb), (0x31cf2b, 0x74, 0xeb),
                           (0x31d031, 0x74, 0xeb), (0x4aae5b, 0x40, 0x60),
                           (0x4b8664, 0x74, 0xeb)]
        for name in ("libcuda.so", "libcuda.so.1"):
            link = custom / relative.parent / name
            assert link.is_symlink() and link.resolve() == custom / relative
        assert (custom / "DEBIAN/triggers").read_bytes() == (official / "DEBIAN/triggers").read_bytes()
        assert (custom / "DEBIAN/shlibs").read_bytes() == (official / "DEBIAN/shlibs").read_bytes()
        assert not any((custom / "DEBIAN" / name).exists()
                       for name in ("preinst", "postinst", "prerm", "postrm"))
        subprocess.run(["md5sum", "--check", "DEBIAN/md5sums"], cwd=custom, check=True,
                       stdout=subprocess.DEVNULL)
        manifest = json.loads((custom / "usr/share/doc/libcuda1-villa/build-manifest.json").read_text())
        assert len(manifest["changes"]) == 5
        assert manifest["libcuda_patched_sha256"] == RELEASE["libcuda_patched_sha256"]
    print("PASS: package identities, dependencies, root ownership, library links, checksums and five reviewed changes")


if __name__ == "__main__":
    main()
