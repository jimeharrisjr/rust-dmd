## Submission

This is a new submission of koopman.dmd 0.2.1, resubmitted after the 0.2.0
pretest. The Debian pretest of 0.2.0 warned in `checking compiled code`
(`exit`, `_exit`, `abort`); 0.2.1 fixes the cause (see "checking compiled
code", below) rather than asking for an exception. (An earlier 0.1.0 was
prepared but never published on CRAN; 0.2.x additionally provides Dynamic
Mode Decomposition with control, `dmdc()`, and its analysis and prediction
methods.)

## Test environments

* win-builder, R devel and R release (x86_64-w64-mingw32)
* macOS 15 (local), R 4.4.1, aarch64
* Ubuntu 24.04 (GitHub Actions), R release and R devel
* macOS 14 (GitHub Actions), R release
* Windows Server 2022 (GitHub Actions), R release

## R CMD check results

| Platform | Result |
|---|---|
| **win-builder, R devel (r90424)** | **0 errors, 0 warnings, 1 note** |
| **win-builder, R release (R 4.6.1)** | **0 errors, 0 warnings, 1 note** |
| macOS 15 (local), R 4.4.1, aarch64 | 0 errors, 0 warnings* |
| Ubuntu 24.04, R release | 0 errors, 1 warning, 0 notes |
| Ubuntu 24.04, R devel   | 0 errors, 1 warning, 0 notes |
| macOS 14, R release     | Status: OK |
| Windows Server 2022, R release | Status: OK |

*The local macOS run reports only environment notes ("unable to verify current
time", HTML-tidy limitations of the system tidy) plus the notes discussed
below.

The win-builder note is `checking CRAN incoming feasibility`, comprising "New
submission" and a list of possibly misspelled words (`Brunton`, `Hankel`,
`Koopman`, `Kutz`, `Mezic`, `Schmid`, `eigenfunction`, `mesochronic`); both
are addressed below. The PDF and HTML manuals and the vignette all built
cleanly on win-builder. The `checking compiled code` warning that the 0.2.0
Debian pretest reported is fixed in 0.2.1 — see the section below.

### Possibly misspelled words

All are correct: `Koopman`, `Hankel`, `Schmid`, `Mezic`, `Proctor`, `Brunton`
and `Kutz` are surnames (Bernard Koopman, Hermann Hankel, Peter Schmid, Igor
Mezic, and the authors of the DMDc paper); `eigenfunction` and `mesochronic`
are standard terms in operator theory and dynamical systems respectively;
`DMDc` is the standard abbreviation for Dynamic Mode Decomposition with
control.

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
and `Authors@R` credits them collectively with a `ctb` role. All 134 bundled
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

### `checking compiled code` (fixed in 0.2.1)

The 0.2.0 Debian pretest warned that the compiled code contains `exit`,
`_exit` and `abort`, attributed to `rust/target/release/libkoopman_dmd_r.a`.
We traced this to the intermediate static archive, not the shared library:
the Rust standard library bundles process-termination and panic runtime
objects into every static archive, but the linker never pulls them into
`koopman.dmd.so` (verified with `nm -D -u` on x86_64 Linux builds under both
rustc 1.95.0 — the pretest machine's toolchain — and current stable: the
linked library references none of these entry points). The warning appeared
because the archive itself was left in the build tree, where
`checking compiled code` now scans the symbol tables of linked static
libraries (PR#18789).

0.2.1 removes the static archive as soon as the shared library has been
linked, following the same pattern as the 'string2path' package cited in
"Using Rust in CRAN packages". The R-facing code never terminates the R
process: all fallible operations return an R condition through 'extendr'.

### Installed size

`checking installed package size` reports roughly 13 Mb on x86_64, almost all of
it the compiled Rust static library under `libs/`. This is the cost of a compiled
Rust backend rather than avoidable payload: the library provides the whole
numerical implementation, including the linear algebra, so the package needs no
external BLAS or LAPACK. The build already strips debug information
(`-C strip=debuginfo` via the release profile). If CRAN would prefer this reduced
further, we are happy to discuss options.
