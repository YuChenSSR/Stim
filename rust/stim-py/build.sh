#!/usr/bin/env bash
# Builds the `stimcore` Python extension and places it where Python can import
# it. With maturin you'd just run `maturin develop`; this avoids that dependency
# by building the cdylib directly and copying it to an importable name.
#
# Usage: rust/stim-py/build.sh [dest_dir]   (dest defaults to this directory)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
workspace="$(cd "$here/.." && pwd)"
dest="${1:-$here}"

cargo build --release -p stim-py --manifest-path "$workspace/Cargo.toml"
cp "$workspace/target/release/libstimcore.so" "$dest/stimcore.so"
echo "Built $dest/stimcore.so"
echo "Import with:  PYTHONPATH=$dest python3 -c 'import stimcore'"
