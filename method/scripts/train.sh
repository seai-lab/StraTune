#!/usr/bin/env bash
# Train StraTune on one dataset with the paper's configuration.
# Usage: method/scripts/train.sh <docvqa|livemath|mind2web|spreadsheetbench> [haiku|opus] [--dry-run]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DS="${1:?dataset}"; SETTING="${2:-haiku}"; shift 2 2>/dev/null || shift $#
HP="$ROOT/method/configs/$DS.json"
if [ "$SETTING" = "opus" ]; then
  export STRATUNE_TARGET_MODEL="${STRATUNE_TARGET_MODEL:-us.anthropic.claude-opus-4-8}"
  export STRATUNE_OPTIMIZER_MODEL="${STRATUNE_OPTIMIZER_MODEL:-us.anthropic.claude-opus-5}"
  [ "$DS" = livemath ] && HP="$ROOT/method/configs/livemath_opus.json"
else
  export STRATUNE_TARGET_MODEL="${STRATUNE_TARGET_MODEL:-global.anthropic.claude-haiku-4-5-20251001-v1:0}"
  export STRATUNE_OPTIMIZER_MODEL="${STRATUNE_OPTIMIZER_MODEL:-global.anthropic.claude-sonnet-4-6}"
fi
export STRATUNE_ROOT="$ROOT" STRATUNE_HP_JSON="$HP"
export STRATUNE_BENCHMARK_DATA="${STRATUNE_BENCHMARK_DATA:-$ROOT/benchmark_data}"
cd "$ROOT" && exec python3 -B -m method.train "$DS" "$@"
