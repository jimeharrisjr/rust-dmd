#!/usr/bin/env bash
#
# Verify that koopman-dmd-r/src/rust/vendor.tar.xz can build the R binding crate with
# no network access, the way CRAN's machines will.
#
# This is the check that matters after tools/vendor-for-cran.sh: the pruning there is
# aggressive, and the only way to know it removed nothing load-bearing is to build
# against the pruned tree with --offline.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUST_DIR="$REPO_ROOT/koopman-dmd-r/src/rust"
VENDOR_TAR="$RUST_DIR/vendor.tar.xz"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

[ -f "$VENDOR_TAR" ] || { echo "!! $VENDOR_TAR missing. Run tools/vendor-for-cran.sh." >&2; exit 1; }

echo "==> Staging a clean copy of the crate"
mkdir -p "$WORK/rust"
cp "$RUST_DIR/Cargo.toml" "$RUST_DIR/Cargo.lock" "$WORK/rust/"
cp -R "$RUST_DIR/src" "$WORK/rust/src"

echo "==> Extracting vendor.tar.xz"
tar xJf "$VENDOR_TAR" -C "$WORK/rust"

echo "==> Configuring cargo for offline vendored sources"
mkdir -p "$WORK/rust/.cargo"
cat > "$WORK/rust/.cargo/config.toml" <<'EOF'
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "vendor"
EOF

# Mirror the CRAN build environment: a CARGO_HOME inside the build tree (never the
# user's home), no network, and at most two jobs.
export CARGO_HOME="$WORK/cargo-home"
mkdir -p "$CARGO_HOME"

echo "==> rustc version (as the CRAN policy asks packages to report)"
rustc --version

echo "==> Building offline with -j 2"
cd "$WORK/rust"
if cargo build --offline --locked --release -j 2 2>&1 | tail -25; then
  echo
  echo "==> OFFLINE BUILD OK -- vendor.tar.xz is self-contained"
else
  echo
  echo "!! OFFLINE BUILD FAILED -- the prune list in vendor-for-cran.sh removed something needed" >&2
  exit 1
fi
