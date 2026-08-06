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

# The layout below deliberately mirrors what R does during R CMD INSTALL: make runs
# with src/ as the working directory and invokes cargo with --manifest-path=./rust/...
# That distinction matters. cargo resolves .cargo/config.toml relative to its working
# directory, not to the manifest, so a config placed beside the manifest is silently
# ignored -- an earlier version of this script built from inside rust/ and therefore
# passed while the real R build failed. Keep this staging faithful to Makevars.
echo "==> Staging a clean copy of the crate in an R-like src/ layout"
mkdir -p "$WORK/src/rust"
cp "$RUST_DIR/Cargo.toml" "$RUST_DIR/Cargo.lock" "$WORK/src/rust/"
cp -R "$RUST_DIR/src" "$WORK/src/rust/src"
cp "$VENDOR_TAR" "$WORK/src/rust/"

cd "$WORK/src"

echo "==> Extracting vendor.tar.xz"
tar xJf ./rust/vendor.tar.xz -C ./rust

# Mirror the CRAN build environment: CARGO_HOME inside the build tree (never the
# user's home), the source-replacement config inside CARGO_HOME so cargo actually
# reads it, no network, and at most two jobs.
export CARGO_HOME="$WORK/src/.cargo"
mkdir -p "$CARGO_HOME"
cat > "$CARGO_HOME/config.toml" <<EOF
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "$WORK/src/rust/vendor"
EOF

echo "==> rustc version (as the CRAN policy asks packages to report)"
rustc --version

echo "==> Building offline with -j 2, from src/ via --manifest-path (as R does)"
if ! cargo build --lib --release --offline --locked -j 2 \
     --manifest-path=./rust/Cargo.toml --target-dir ./rust/target 2>&1 | tail -25; then
  echo
  echo "!! OFFLINE BUILD FAILED -- the prune list in vendor-for-cran.sh removed something needed" >&2
  exit 1
fi
echo "==> host build OK"

# faer enables private-gemm-x86 only on x86_64, and that subtree (spindle,
# atomic-wait, ...) is invisible to a build on an aarch64 host. Checking an x86_64
# target as well is what catches pruning mistakes in those crates -- a host-only
# check on Apple Silicon once passed while CI failed on exactly this.
X86_TARGET=""
for t in x86_64-apple-darwin x86_64-unknown-linux-gnu; do
  if rustc --print target-libdir --target "$t" >/dev/null 2>&1; then X86_TARGET="$t"; break; fi
done

if [ -z "$X86_TARGET" ]; then
  echo "!! WARNING: no x86_64 target installed, so the private-gemm-x86 subtree" >&2
  echo "!! (spindle, atomic-wait, ...) was NOT exercised. Install one with:" >&2
  echo "!!     rustup target add x86_64-apple-darwin" >&2
else
  echo "==> Cross-checking the x86_64-only dependency subtree ($X86_TARGET)"
  if cargo check --lib --release --offline --locked -j 2 --target "$X86_TARGET" \
       --manifest-path=./rust/Cargo.toml --target-dir ./rust/target-x86 2>&1 | tail -20; then
    echo "==> x86_64 subtree OK"
  else
    echo "!! x86_64 CHECK FAILED -- pruning broke a crate reachable only on x86_64" >&2
    exit 1
  fi
fi

echo
echo "==> OFFLINE BUILD OK -- vendor.tar.xz is self-contained"
