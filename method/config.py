"""Datasets, splits, and training budgets used in the paper."""


DATASETS = ["docvqa", "livemath", "mind2web", "spreadsheetbench"]

# Directory of each dataset under ${STRATUNE_BENCHMARK_DATA}; every dataset has
# train/ and test/ subdirectories with a task_ids.json listing the split.
BENCHMARK_DIRS = {"docvqa": "DocVQA/dev", "livemath": "LiveMathematicianBench",
                  "mind2web": "Mind2Web", "spreadsheetbench": "SpreadsheetBench"}
SPLIT_SIZES = {"docvqa": {"train": 500, "test": 1000}, "livemath": {"train": 397, "test": 211},
               "mind2web": {"train": 800, "test": 252}, "spreadsheetbench": {"train": 629, "test": 280}}
TRAIN_SPLITS = {ds: f"train{n['train']}" for ds, n in SPLIT_SIZES.items()}
TEST_SPLITS = {ds: f"test{n['test']}" for ds, n in SPLIT_SIZES.items()}

# Stratification fields (data/strata/<dataset>/strata_meta_train.json) used for
# the training batches and the stratified part of the screening sets.
STRATA_FIELDS = {"docvqa": ["qlen_bucket"], "livemath": ["month"],
                 "mind2web": ["website", "steps_bucket"],
                 "spreadsheetbench": ["instruction_type", "ilen_bucket"]}

# Training budget in target-LLM executions: BUDGET_MULTIPLIER x |D_train|
# (3000 / 2382 / 4800 / 3774 for DocVQA / LiveMath / Mind2Web / SpreadsheetBench).
BUDGET_MULTIPLIER = 6


def check_dataset(dataset: str):
    if dataset not in DATASETS:
        raise RuntimeError(f"unknown dataset {dataset!r}; choose from {DATASETS}")


# Model ids (Amazon Bedrock). The paper's Haiku/Sonnet setting is the default;
# the Opus setting is selected by method/scripts/train.sh.
import os

OPTIMIZER_MODEL = os.environ.get("STRATUNE_OPTIMIZER_MODEL", "global.anthropic.claude-sonnet-4-6")
TARGET_MODEL = os.environ.get("STRATUNE_TARGET_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0")


def _truncate(s, n=2000):
    s = str(s)
    return s if len(s) <= n else s[:n] + f"...[+{len(s)-n} chars]"
