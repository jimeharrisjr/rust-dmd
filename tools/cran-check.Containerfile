# A Debian + R + Rust image for running `R CMD check --as-cran` on koopman.dmd
# locally, the way CRAN's Linux machines do.
#
# Two reasons this exists:
#
#  1. The maintainer's macOS R installation cannot compile packages at all (its
#     toolchain resolves an iOS SDK), so there is no local way to check the package.
#  2. The GitHub runners have neither LaTeX nor HTML Tidy, so CI has to pass
#     --no-manual and skips HTML validation. CRAN has both. This image installs
#     them, so the check here is stricter than CI rather than weaker.
#
# Build and run via tools/cran-check.sh.

FROM docker.io/library/r-base:latest

ENV DEBIAN_FRONTEND=noninteractive

# Build toolchain, plus the pieces R CMD check needs for the manual and HTML
# validation that the GitHub runners lack.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      pkg-config \
      xz-utils \
      qpdf \
      tidy \
      texinfo \
      # rmarkdown shells out to pandoc to build the vignette; without it
      # R CMD build fails at "creating vignettes".
      pandoc \
      # fs (>= 2.1.0) needs libuv headers, and pkgload -> testthat depend on fs.
      # Without this the R package installs below fail, which is exactly the
      # breakage that took down the R CI job earlier in this project.
      libuv1-dev \
      libssl-dev \
      libxml2-dev \
      zlib1g-dev \
      texlive-latex-base \
      texlive-latex-recommended \
      texlive-fonts-recommended \
      texlive-plain-generic \
      # R's Rd-to-PDF manual uses the inconsolata font; without it the PDF check
      # fails with "File `inconsolata.sty' not found".
      texlive-fonts-extra \
      # Provides checkbashisms, which `checking top-level files` wants in order to
      # vet configure/configure.win for non-POSIX shell constructs.
      devscripts \
    && rm -rf /var/lib/apt/lists/*

# Rust via rustup. CRAN's own machines use distribution packages, but those lag
# well behind the 1.85 this package requires; rustup is what a user on an older
# distribution would be told to do by ./configure.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# install.packages() only warns on failure, so verify explicitly -- otherwise the
# image builds "successfully" with the check dependencies missing.
RUN Rscript -e 'pkgs <- c("testthat", "knitr", "rmarkdown"); \
      install.packages(pkgs, repos = "https://cloud.r-project.org", Ncpus = 4); \
      missing <- pkgs[!pkgs %in% rownames(installed.packages())]; \
      if (length(missing)) stop("failed to install: ", paste(missing, collapse = ", "))'

WORKDIR /work
CMD ["/bin/bash"]
