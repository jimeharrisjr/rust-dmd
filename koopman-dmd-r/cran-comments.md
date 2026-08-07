## Submission

This is a new submission of koopman.dmd 0.1.0.

## Test environments

* win-builder, R devel and R release (x86_64-w64-mingw32)
* Debian GNU/Linux forky/sid, R 4.6.1, x86_64 (container, full check with LaTeX)
* Debian GNU/Linux forky/sid, R 4.6.1, aarch64 (container, full check with LaTeX)
* Ubuntu 24.04 (GitHub Actions), R release and R devel
* macOS 14 (GitHub Actions), R release
* Windows Server 2022 (GitHub Actions), R release

## R CMD check results

| Platform | Result |
|---|---|
| **win-builder, R devel** | **0 errors, 0 warnings, 1 note** |
| **win-builder, R release** | **0 errors, 0 warnings, 1 note** |
| Debian, R 4.6.1, x86_64 | 0 errors, 1 warning, 0 notes |
| Debian, R 4.6.1, aarch64 | 0 errors, 1 warning, 1 note |
| Ubuntu 24.04, R release | 0 errors, 1 warning, 0 notes |
| Ubuntu 24.04, R devel   | 0 errors, 1 warning, 0 notes |
| macOS 14, R release     | Status: OK |
| Windows Server 2022, R release | Status: OK |

The win-builder note is `checking CRAN incoming feasibility`, comprising "New
submission" and a list of possibly misspelled words; both are addressed below.
`checking compiled code` was **OK** on win-builder, and the PDF manual built
cleanly. The warning of that name appears on Linux only and is explained further
down; we do not believe it is removable from a Rust package, but if you would
prefer it handled differently, please say so and we will follow your guidance.

### Possibly misspelled words

All are correct: `Koopman`, `Hankel`, `Schmid` and `Mezic` are surnames (Bernard
Koopman, Hermann Hankel, Peter Schmid, Igor Mezic); `eigenfunction` and
`mesochronic` are standard terms in operator theory and dynamical systems
respectively.

### Note on aarch64 only

The additional note on aarch64 is `Compilation used the following non-portable
flag(s): '-mbranch-protection=standard'`. That flag comes from R's own CFLAGS as
configured by Debian on arm64, not from this package's `Makevars`, and it does not
appear on x86_64.

## Notes for the reviewer

### Bundled Rust sources

The package uses a Rust backend via 'extendr'. Following the guidance in
"Using Rust in CRAN packages", the complete dependency tree is bundled in
`src/rust/vendor.tar.xz` (xz-compressed, approximately 4 MB) rather than
downloaded at install time. The build runs `cargo build --offline --locked`
against those vendored sources and makes no network access whatsoever.

`CARGO_HOME` is set to a directory inside the build tree and removed afterwards,
so nothing is written to the user's home directory. The build is limited to two
jobs (`-j 2`) in line with the CRAN policy on parallelism.

The vendored tree is pruned to stay within the tarball size guidance: crates that
cannot participate in the build (a `cfg(loom)`-only dependency and its subtree)
and Windows import libraries for targets R no longer supports (32-bit, and the
MSVC toolchains) are reduced to stubs. `tools/check-vendor.sh` in the source
repository verifies that the pruned tree still builds offline.

Authorship and licensing for every bundled crate is recorded in `inst/AUTHORS`,
and `Authors@R` credits them collectively with a `ctb` role. All 133 bundled
crates carry permissive licences (MIT, Apache-2.0, BSD-2-Clause, Zlib,
Unlicense, Unicode-3.0), each compatible with this package's MIT licence.

### Rust version requirement

`SystemRequirements` declares `rustc (>= 1.85)`. This floor is not set by the
package's own code but by a transitive dependency of the linear algebra crate
`faer`, which requires Rust edition 2024.

We investigated lowering it. Older `faer` releases would reach Rust 1.81, and an
extensive rewrite against `faer` 0.19 would reach 1.71, but none of those reach
the toolchain in Debian oldstable (1.63), so the rewrite would buy nothing in
practice. Debian stable (trixie) ships rustc 1.85.0, so current distribution
toolchains satisfy the requirement.

`configure` and `configure.win` check for `cargo` and `rustc` (including
personal installs under `~/.cargo/bin`, which are often absent from `PATH` in
non-interactive builds), verify the version, and report it into the installation
log before compilation begins. If the toolchain is missing or too old, they emit
installation instructions and stop; they never attempt to install anything.

### `checking compiled code` note

On Linux the check reports that the compiled library contains `exit`, `_exit` and
`abort`. These symbols come from the Rust standard library's panic and
process-abort machinery, which is linked into every Rust static library. They are
not called by this package's own code. The same check reports OK on win-builder
and on macOS, so the difference is in which platforms' checks surface the symbols
rather than in what the library contains.

The R-facing code never terminates the R process: all fallible operations return
an R condition through 'extendr', and the package installs no panic handler that
aborts. We are not aware of a way to prevent a Rust static library from carrying
these symbols, but if you know of one, or would prefer this handled differently,
we will follow your guidance.

### Installed size

`checking installed package size` reports roughly 13 Mb on x86_64, almost all of
it the compiled Rust static library under `libs/`. This is the cost of a compiled
Rust backend rather than avoidable payload: the library provides the whole
numerical implementation, including the linear algebra, so the package needs no
external BLAS or LAPACK. The build already strips debug information
(`-C strip=debuginfo` via the release profile). If CRAN would prefer this reduced
further, we are happy to discuss options.
