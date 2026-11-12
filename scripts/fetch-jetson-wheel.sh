#!/bin/sh
# Fetch one verified Jetson wheel from an ordered, space-separated URL list.
# This tiny dependency-free helper is copied into Dockerfile.jetson and is
# deliberately independently testable with a mocked curl executable.
set -eu

output=${1:?output path is required}
urls=${2:?at least one wheel URL is required}

for url in $urls; do
  filename=${url%%\#*}
  echo "Downloading ${filename##*/}"
  if curl --fail --location --retry 2 --retry-all-errors --output "$output" "$url"; then
    exit 0
  fi
  rm -f "$output"
  echo "Wheel download failed; trying the next verified source." >&2
done

echo "Could not download a Jetson wheel from any configured source." >&2
exit 1
