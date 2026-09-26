#!/usr/bin/env bash
# Evaluate a skill on the official test split.
# Usage: method/scripts/evaluate.sh --run-id <training_run_id> [--workers N]
#        method/scripts/evaluate.sh --skill data/skills/haiku/livemath.txt --dataset livemath [--workers N]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export STRATUNE_ROOT="$ROOT"
export STRATUNE_BENCHMARK_DATA="${STRATUNE_BENCHMARK_DATA:-$ROOT/benchmark_data}"
cd "$ROOT" && exec python3 -B -m method.evaluate "$@"
