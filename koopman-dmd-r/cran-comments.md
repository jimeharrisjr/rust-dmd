## Submission

This is a new submission of koopman.dmd 0.1.0.

## Test environments

* Ubuntu 24.04 (GitHub Actions), R release and R devel
* macOS 14 (GitHub Actions), R release
* Windows Server 2022 (GitHub Actions), R release
* win-builder, R devel and R release

## R CMD check results

0 errors | 0 warnings | 0 notes

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

The check reports that the compiled library contains `exit`, `_exit` and `abort`.
These come from the Rust standard library's panic and process-abort machinery,
which is linked into every Rust static library; they are not called by this
package's own code, and no Rust package can avoid exporting them. The R-facing
code never terminates the R process: all fallible operations return an R
condition through 'extendr', and the package sets no panic handler that aborts.
