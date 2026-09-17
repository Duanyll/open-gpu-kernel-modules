#!/usr/bin/env python3
"""Build the reviewed libcuda1-villa package from an official NVIDIA deb, offline."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_control(path):
    fields = {}
    key = None
    for line in path.read_text().splitlines():
        if line.startswith((" ", "\t")) and key:
            fields[key] += "\n" + line
        elif line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    return fields


def build(source, output, release):
    require(not output.exists(), f"output already exists: {output}")
    require(digest(source) == release["libcuda_deb_sha256"], "official deb SHA-256 mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="libcuda-villa-", dir=output.parent) as temporary:
        work = Path(temporary)
        stage = work / "root"
        subprocess.run(["dpkg-deb", "--raw-extract", str(source), str(stage)], check=True)
        stage.chmod(0o755)
        control = stage / "DEBIAN"
        require({p.name for p in control.iterdir()} <= {"control", "md5sums", "shlibs", "triggers", "symbols"},
                "unexpected upstream maintainer scripts or control files")
        fields = read_control(control / "control")
        require((fields["Package"], fields["Version"], fields["Architecture"]) ==
                ("libcuda1", release["upstream_version"], "amd64"), "unexpected upstream package identity")
        library = stage / "usr/lib/x86_64-linux-gnu" / f"libcuda.so.{release['driver_version']}"
        require(digest(library) == release["libcuda_original_sha256"], "original libcuda SHA-256 mismatch")
        require(release["patch_set"] == "all", "this package requires the reviewed P2P and GDR patch set")

        spec = importlib.util.spec_from_file_location("libcuda_patcher", ROOT / "patch-libcuda-p2p.py")
        patcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patcher)
        original = library.read_bytes()
        plan = patcher.plan_patches(original, patcher.P2P_PATCHES + patcher.GDR_PATCHES)
        patched = bytearray(original)
        for _, offset, before, replacement in plan:
            require(before != replacement, "input library already contains a selected patch")
            patched[offset:offset + len(replacement)] = replacement
        require(hashlib.sha256(patched).hexdigest() == release["libcuda_patched_sha256"],
                "patched libcuda SHA-256 mismatch")
        library.write_bytes(patched)

        fields.update({
            "Package": "libcuda1-villa",
            "Version": release["package_version"],
            "Maintainer": "duanyll <duanyll@outlook.com>",
            "Provides": f"libcuda1 (= {release['upstream_version']}), libcuda.so.1 (= {release['driver_version']})",
            "Conflicts": "libcuda1",
            "Replaces": "libcuda1",
            "Description": "NVIDIA CUDA driver library with P2P and DMA-BUF GDR patches\n"
                           " Includes the reviewed x86-64 P2P and DMA-BUF GDR patches.\n"
                           " Requires matching NVIDIA kernel modules and userspace.\n"
                           " Unofficial; not affiliated with NVIDIA.",
        })
        # Source identifies our packaging, not an unmodified NVIDIA source package.
        fields.pop("Source", None)
        docs = stage / "usr/share/doc/libcuda1-villa"
        (stage / "usr/share/doc/libcuda1").rename(docs)
        manifest = dict(release)
        manifest["patcher_sha256"] = digest(ROOT / "patch-libcuda-p2p.py")
        manifest["changes"] = [dict(name=name, offset=hex(offset), before=before.hex(), after=after.hex())
                               for name, offset, before, after in plan]
        if (ROOT / ".git").exists():
            manifest["source_commit"] = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        (docs / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        fields["Installed-Size"] = str(sum(p.stat().st_size for p in stage.rglob("*")
                                          if p.is_file() and not p.is_symlink()
                                          and control not in p.parents) // 1024 + 1)
        (control / "control").write_text("".join(f"{key}: {value}\n" for key, value in fields.items()))
        sums = []
        for path in sorted(stage.rglob("*")):
            if path.is_file() and not path.is_symlink() and control not in path.parents:
                sums.append(f"{hashlib.md5(path.read_bytes()).hexdigest()}  {path.relative_to(stage)}\n")
        (control / "md5sums").write_text("".join(sums))
        artifact = work / "package.deb"
        subprocess.run(["dpkg-deb", "--root-owner-group", "--build", str(stage), str(artifact)], check=True)
        # Exclusive creation keeps an existing release artifact intact.
        with output.open("xb") as target, artifact.open("rb") as stream:
            shutil.copyfileobj(stream, target)
        output.chmod(0o644)
    print(f"built {output} (sha256 {digest(output)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_deb", type=Path, help="official libcuda1 deb from packaging/release.json")
    parser.add_argument("output_deb", type=Path, nargs="?", help="new output file")
    args = parser.parse_args()
    os.umask(0o022)
    release = json.loads((ROOT / "packaging/release.json").read_text())
    output = args.output_deb or Path(f"libcuda1-villa_{release['package_version']}_amd64.deb")
    try:
        build(args.input_deb.resolve(), output.resolve(), release)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"build failed: {error}\n")


if __name__ == "__main__":
    main()
