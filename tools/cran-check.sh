#!/usr/bin/env bash
#
# Run `R CMD check --as-cran` on koopman.dmd inside a Debian container, using
# Apple's `container` runtime.
#
# Why this exists: the maintainer's macOS R cannot compile packages (its toolchain
# resolves an iOS SDK), so there is otherwise no way to check the package locally.
# The container also carries LaTeX and HTML Tidy, which the GitHub runners lack, so
# this check is stricter than CI -- it builds the PDF manual and validates the HTML
# rather than skipping both.
#
#   tools/cran-check.sh              # native arch (fast)
#   tools/cran-check.sh --arch amd64 # x86_64, matching CRAN's Linux machines
#
# Note that on Apple Silicon, --arch amd64 is emulated and considerably slower, but
# it is the only way to exercise the x86_64-only parts of the vendored Rust tree
# (private-gemm-x86 -> spindle -> atomic-wait) in a full package build.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_BASE="koopman-cran-check"
ARCH="$(uname -m)"
[ "$ARCH" = "x86_64" ] && ARCH="amd64"
[ "$ARCH" = "aarch64" ] && ARCH="arm64"

while [ $# -gt 0 ]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! command -v container >/dev/null 2>&1; then
  echo "!! Apple's 'container' CLI is not installed (requires macOS 26+)." >&2
  echo "!! See https://github.com/apple/container" >&2
  exit 1
fi

container system start >/dev/null 2>&1 || true

# Tag per architecture. A single tag would let an arm64 image satisfy a request for
# amd64 (or vice versa) and silently check the wrong platform.
IMAGE="${IMAGE_BASE}:${ARCH}"

if ! container image list 2>/dev/null | grep -q "${IMAGE_BASE}[[:space:]]*${ARCH}\b"; then
  echo "==> Building $IMAGE; this takes a few minutes the first time"
  container build --arch "$ARCH" -t "$IMAGE" \
    -f "$REPO_ROOT/tools/cran-check.Containerfile" "$REPO_ROOT"
fi

echo "==> Running R CMD check --as-cran in $IMAGE ($ARCH)"

# The container's default allocation is too small: rustc is OOM-killed (SIGKILL)
# compiling the codegen-heavy nano-gemm crates. These are generous but well within
# a typical developer machine; lower them if your host is smaller.
MEM="${CRAN_CHECK_MEMORY:-8g}"
CPUS="${CRAN_CHECK_CPUS:-4}"
echo "    (memory $MEM, cpus $CPUS -- override with CRAN_CHECK_MEMORY / CRAN_CHECK_CPUS)"

# The package directory is mounted read-only and copied inside, so the build never
# writes into the working tree -- R CMD build would otherwise leave artefacts and
# the Rust target directory behind.
# The built tarball is the artefact to submit to CRAN and upload to win-builder, so
# copy it out of the (--rm) container rather than losing it. Local `R CMD build`
# cannot produce this: building the vignette installs the package first, which needs
# a working compiler.
DIST_DIR="$REPO_ROOT/dist"
mkdir -p "$DIST_DIR"

container run --rm --arch "$ARCH" --memory "$MEM" --cpus "$CPUS" \
  --volume "$REPO_ROOT/koopman-dmd-r:/src:ro" \
  --volume "$DIST_DIR:/out" \
  "$IMAGE" bash -euo pipefail -c '
    export PATH="/root/.cargo/bin:$PATH"
    echo "== toolchain =="
    R --version | head -1
    rustc --version
    cargo --version
    echo

    cp -R /src /work/pkg
    cd /work
    # Drop anything the working tree may carry that must not enter the tarball.
    rm -rf pkg/src/rust/target pkg/src/rust/vendor pkg/src/*.o pkg/src/*.so pkg/check

    echo "== R CMD build =="
    R CMD build pkg
    TARBALL=$(ls -1 koopman.dmd_*.tar.gz | head -1)
    echo "built: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
    cp "$TARBALL" /out/
    echo "copied to dist/$TARBALL"
    echo

    echo "== R CMD check --as-cran =="
    # NOT_CRAN unset so the vendored offline path is exercised exactly as CRAN will.
    _R_CHECK_CRAN_INCOMING_REMOTE_=false \
      R CMD check --as-cran "$TARBALL" || true

    echo
    echo "===================== 00check.log ====================="
    sed -n "/^\* checking/,\$p" koopman.dmd.Rcheck/00check.log | grep -vE "\.\.\. OK$" || true
    echo "======================================================="
    tail -5 koopman.dmd.Rcheck/00check.log
  '
