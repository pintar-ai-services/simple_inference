#!/bin/sh
# Fetch one verified Jetson wheel from an ordered, space-separated URL list.
# This tiny dependency-free helper is copied into Dockerfile.jetson and is
# deliberately independently testable with a mocked curl executable.
set -eu

output_dir=${1:?output directory is required}
filename=${2:?wheel filename is required}
urls=${3:?at least one wheel URL is required}
output="${output_dir}/${filename}"

case "$filename" in
  *.whl) ;;
  *) echo "wheel filename must end in .whl: $filename" >&2; exit 2 ;;
esac

for url in $urls; do
  echo "Downloading ${filename}"
  if curl --fail --location --retry 2 --retry-all-errors --output "$output" "$url"; then
    exit 0
  fi
  rm -f "$output"
  echo "Wheel download failed; trying the next verified source." >&2
done

echo "Could not download a Jetson wheel from any configured source." >&2
exit 1
