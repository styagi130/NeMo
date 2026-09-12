#!/usr/bin/env bash
set -euo pipefail

# The Python supervisor owns its private foreground daemon and cleans it up.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$script_dir/launch_h100.py" --private-mps "$@"
