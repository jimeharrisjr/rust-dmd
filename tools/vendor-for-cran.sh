#!/usr/bin/env bash
#
# Produce koopman-dmd-r/src/rust/vendor.tar.xz: a self-contained, pruned copy of the
# Rust dependency tree for the CRAN package.
#
# CRAN build machines are offline, so every Rust source must ship inside the package
# tarball (see https://cran.r-project.org/web/packages/using_rust.html). A plain
# `cargo vendor` produces ~14 MB compressed, well over CRAN's ~5 MB guidance. The
# pruning below brings it to roughly 4 MB.
#
# What dominates the raw size, and why each prune is safe:
#
#   * faer enables private-gemm-x86 on x86_64, which pulls spindle -> loom ->
#     tracing-subscriber -> nu-ansi-term. `loom` is a cfg(loom)-only dependency that
#     can never compile in a normal build, yet it drags in ~4 MB of compressed
#     windows-sys bindings. The whole chain is dead weight.
#   * atomic-wait genuinely needs windows-sys on Windows, but only for
#     Win32::System::Threading and Win32::Foundation. windows-sys is a modular tree
#     whose unused subtrees are never referenced when their features are off.
#   * CRAN's Windows builds use the GNU toolchain (Rtools) on x86_64 only; R dropped
#     32-bit Windows in 4.2. The msvc and i686 import libraries are unreachable.
#
# Re-run this whenever the Rust dependencies change, then commit the result.
# Verify afterwards with tools/check-vendor.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUST_DIR="$REPO_ROOT/koopman-dmd-r/src/rust"
OUT="$RUST_DIR/vendor.tar.xz"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# CRAN's guidance is ~5 MB for the whole package; leave room for R sources and docs.
MAX_BYTES=$((5 * 1024 * 1024))

echo "==> Vendoring dependencies from $RUST_DIR"
cd "$RUST_DIR"

if [ ! -f Cargo.lock ]; then
  echo "!! Cargo.lock missing. Run 'cargo generate-lockfile' in $RUST_DIR first." >&2
  exit 1
fi

cargo vendor --versioned-dirs --locked "$WORK/vendor" > "$WORK/config-snippet.toml"

RAW_SIZE=$(du -sm "$WORK/vendor" | cut -f1)
echo "    raw vendored size: ${RAW_SIZE} MB"

cd "$WORK/vendor"

# Cargo resolves the whole lockfile graph before compiling anything, and a vendored
# source directory must contain EVERY locked package -- including ones that can never
# be compiled, like cfg(loom)-only dependencies. So these crates cannot be deleted;
# they are reduced to a manifest plus an empty lib, which keeps resolution happy while
# dropping the payload. Anything stubbed here must be genuinely unreachable at compile
# time; tools/check-vendor.sh is what proves that.
stub_crate() {
  local dir="$1"
  [ -d "$dir" ] || return 0
  local libpath
  # Honour an explicit [lib] path if the manifest sets one, else the default.
  libpath=$(python3 - "$dir/Cargo.toml" <<'PY'
import re, sys
try:
    txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
except OSError:
    print("src/lib.rs"); raise SystemExit
m = re.search(r'^\s*\[lib\]', txt, re.M)
path = "src/lib.rs"
if m:
    tail = txt[m.end():]
    pm = re.search(r'^\s*path\s*=\s*"([^"]+)"', tail.split("\n[")[0], re.M)
    if pm:
        path = pm.group(1)
print(path)
PY
)
  find "$dir" -mindepth 1 -maxdepth 1 \
       ! -name Cargo.toml ! -name .cargo-checksum.json -exec rm -rf {} + 2>/dev/null || true
  mkdir -p "$dir/$(dirname "$libpath")"
  : > "$dir/$libpath"
  # A build script referenced by the manifest must exist even if never run.
  if grep -qE '^\s*build\s*=\s*"' "$dir/Cargo.toml" 2>/dev/null; then
    local bs
    bs=$(sed -n 's/^[[:space:]]*build[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$dir/Cargo.toml" | head -1)
    [ -n "$bs" ] && { mkdir -p "$dir/$(dirname "$bs")"; printf 'fn main() {}\n' > "$dir/$bs"; }
  fi
}

echo "==> Stubbing crates that can never compile in a CRAN build"
# loom is a cfg(loom)-only dependency of spindle; tracing, tracing-subscriber,
# nu-ansi-term and windows-sys 0.61 reach the graph only through it.
for d in loom-* tracing-* nu-ansi-term-* sharded-slab-* thread_local-* \
         matchers-* overload-* regex-automata-0.1.* regex-syntax-0.6.* \
         windows-sys-0.61.* windows-targets-0.53.* windows-link-* windows_*-0.53.* \
         winapi-util-* walkdir-* same-file-* generator-* scoped-tls-* pin-project-lite-*; do
  stub_crate "$d"
done

echo "==> Stubbing unreachable Windows import libraries"
# CRAN's Windows builds are x86_64 GNU (Rtools) only -- R dropped 32-bit in 4.2.
# windows_x86_64_gnu keeps its real import library; the rest are unreachable.
for d in windows_i686_* windows_*_msvc-* windows_*_gnullvm-* windows_aarch64_*; do
  stub_crate "$d"
done

echo "==> Pruning unused windows-sys module subtrees"
for ws in windows-sys-0.4*; do
  [ -d "$ws" ] || continue
  W="$ws/src/Windows/Win32"
  [ -d "$W" ] || continue
  # atomic-wait imports only Win32::System::{Threading,WindowsProgramming} and Foundation.
  for d in "$W"/*; do
    b="$(basename "$d")"
    [ "$b" = "System" ] || [ "$b" = "Foundation" ] || rm -rf "$d"
  done
  for d in "$W"/System/*; do
    b="$(basename "$d")"
    [ "$b" = "Threading" ] || [ "$b" = "WindowsProgramming" ] || rm -rf "$d"
  done
done

echo "==> Stripping test, benchmark, example and doc payloads"
find . -mindepth 2 -maxdepth 3 -type d \
     \( -name tests -o -name benches -o -name examples -o -name fuzz \
        -o -name .github -o -name docs -o -name book -o -name ci \) \
     -exec rm -rf {} + 2>/dev/null || true
# Markdown is deliberately NOT deleted. Crates commonly pull their README into the
# crate docs with #![doc = include_str!("../README.md")], and removing it breaks the
# build -- atomic-wait does exactly this. The saving is negligible next to the risk.
find . -mindepth 2 -maxdepth 2 -type f \
     \( -name '*.png' -o -name '*.jpg' -o -name '*.gif' \
        -o -name '*.svg' -o -name '*.yml' -o -name '*.yaml' \) \
     -delete 2>/dev/null || true

# cargo verifies each vendored crate against .cargo-checksum.json. Pruning changes
# file hashes, so clear the per-file map while keeping the package-level checksum;
# cargo accepts an empty "files" object and still validates the crate itself.
echo "==> Rewriting vendored checksum manifests"
find . -name .cargo-checksum.json -print0 | while IFS= read -r -d '' f; do
  python3 - "$f" <<'PY'
import json, sys
p = sys.argv[1]
with open(p) as fh:
    data = json.load(fh)
data["files"] = {}
with open(p, "w") as fh:
    json.dump(data, fh)
PY
done

PRUNED_SIZE=$(du -sm . | cut -f1)
echo "    pruned size: ${PRUNED_SIZE} MB (from ${RAW_SIZE} MB)"

echo "==> Compressing to $OUT"
cd "$WORK"
tar cf - vendor | xz -9 -T0 > "$OUT"

FINAL=$(wc -c < "$OUT" | tr -d ' ')
printf "    vendor.tar.xz: %.1f MB\n" "$(echo "$FINAL" | awk '{print $1/1048576}')"

if [ "$FINAL" -gt "$MAX_BYTES" ]; then
  printf "!! vendor.tar.xz is %.1f MB, over the %d MB budget.\n" \
    "$(echo "$FINAL" | awk '{print $1/1048576}')" "$((MAX_BYTES / 1024 / 1024))" >&2
  echo "!! Investigate with: du -sm vendor/* | sort -rn | head" >&2
  exit 1
fi

echo "==> Done. Verify the offline build with tools/check-vendor.sh"
