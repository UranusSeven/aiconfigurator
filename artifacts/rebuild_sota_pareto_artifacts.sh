#!/usr/bin/env bash
set -euo pipefail

# Rebuild the compact CSV and PNG artifacts from existing sweep directories.
# Run from the repository root after the GB300/DGX-B300 sweep commands have
# produced their pareto.csv files under artifacts/*_pareto/.

uv run python artifacts/gb300_pareto/build_pareto_artifacts.py
uv run python artifacts/b300_pareto/build_pareto_artifacts.py
