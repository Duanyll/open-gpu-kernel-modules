#!/usr/bin/env bash
# Build the nvidia-open-p2p-dkms .deb from this open-gpu-kernel-modules tree.
#
# Version-agnostic: reads NVIDIA_VERSION from version.mk, so the same packaging works on every
# <ver>-p2p[-48g] branch. Offline: the patched source is vendored into the .deb; DKMS builds it on
# the target with no network. Drop-in replacement for the CUDA-repo nvidia-kernel-open-dkms.
#
# Only the root dkms.conf differs from the stock package: the tree ships kernel-open/dkms.conf as an
# nvidia-installer template (__PLACEHOLDER__s DKMS can't consume); we drop it and generate a
# processed one here. All other build glue is identical to stock.
#
# Needs dpkg-deb (+ git only when run inside a git checkout). Run on any Debian box.
# Usage: build-deb.sh [OUTPUT_DEB]
set -euo pipefail
umask 022

HERE="$(cd "$(dirname "$0")" && pwd)"
SRCTREE="$(cd "$HERE/../.." && pwd)"   # repo root (this packaging lives at packaging/dkms/)

PKG=nvidia-open-p2p-dkms      # deb package name
DKMS_NAME=nvidia-open-p2p     # dkms module tree name (/usr/src/<DKMS_NAME>-<VER>)
DEB_REV=1
VER="$(awk -F= '/^NVIDIA_VERSION[[:space:]]*=/{gsub(/[[:space:]]/,"",$2);print $2;exit}' "$SRCTREE/version.mk")"
[ -n "$VER" ] || { echo "cannot read NVIDIA_VERSION from $SRCTREE/version.mk" >&2; exit 1; }

STAGE="$(mktemp -d)"
chmod 0755 "$STAGE"   # mktemp -d is 0700; the deb's root './' member inherits it and unpacking
                      # would chmod / to 0700. Keep the staging root — hence / — world-traversable.
trap 'rm -rf "$STAGE"' EXIT

SRC="$STAGE/usr/src/${DKMS_NAME}-${VER}"
install -d "$SRC" "$STAGE/DEBIAN"

# Vendor only the build inputs (explicit keep-list, mirroring the stock deb's /usr/src layout).
KEEP=(Makefile nv-compiler.sh utils.mk version.mk kernel-open src)
if git -C "$SRCTREE" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$SRCTREE" archive HEAD -- "${KEEP[@]}" | tar -x -C "$SRC"
else
  tar -C "$SRCTREE" -c "${KEEP[@]}" | tar -x -C "$SRC"
fi
find "$SRC" -type d -exec chmod 0755 {} +
find "$SRC" -type f -perm /111 -exec chmod 0755 {} +
find "$SRC" -type f ! -perm /111 -exec chmod 0644 {} +
rm -f "$SRC/kernel-open/dkms.conf"   # the unusable installer template

# Generate versioned files from templates (@VER@ / @PKG@ / @DKMS_NAME@).
gen() { sed -e "s/@VER@/${VER}/g" -e "s/@PKG@/${PKG}/g" -e "s/@DKMS_NAME@/${DKMS_NAME}/g" "$1"; }
gen "$HERE/dkms.conf.in" > "$SRC/dkms.conf"
gen "$HERE/control.in"   > "$STAGE/DEBIAN/control"
gen "$HERE/postinst.in"  > "$STAGE/DEBIAN/postinst"; chmod 0755 "$STAGE/DEBIAN/postinst"
gen "$HERE/prerm.in"     > "$STAGE/DEBIAN/prerm";    chmod 0755 "$STAGE/DEBIAN/prerm"

OUT="${1:-$HERE/${PKG}_${VER}-${DEB_REV}_all.deb}"
dpkg-deb --root-owner-group --build "$STAGE" "$OUT"
echo "built $OUT"
