#!/usr/bin/env bash
# Download public talk captions (OPUS TED2020 corpus, one caption line per
# line) for the benchmark languages into benchmarks/data/cc/<lang>.txt.gz.
#
#   usage: scripts/pull_cc_data.sh [lang ...]     (default: en ru hy)
#
# The files are cached (skipped when already present). benchmarks/data/cc/ is
# git-ignored; the checked-in benchmarks/data/samples/ subset is used by unit
# tests and as a fallback when the full corpus has not been pulled.
set -euo pipefail
cd "$(dirname "$0")/../benchmarks/data"
mkdir -p cc

for lang in "$@"; do
  out="cc/${lang}.txt.gz"
  if [ -s "${out}" ]; then
    echo "exists: ${out}"
    continue
  fi
  url="https://object.pouta.csc.fi/OPUS-TED2020/v1/mono/${lang}.txt.gz"
  echo "fetching ${url}"
  curl -fsSL -o "${out}" "${url}"
done

for f in cc/*.txt.gz; do
  echo "${f}: $(zcat "${f}" | wc -l) lines"
done
