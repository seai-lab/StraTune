# StraTune

**Adaptive Selection of Revision Operators for Self-Evolving LLM Skills**

[Paper (arXiv:XXXX.XXXXX)](https://arxiv.org/abs/XXXX.XXXXX) · [Benchmark data](https://huggingface.co/datasets/PingL/StraTune_dataset)

StraTune learns a textual skill for a frozen target LLM with a frozen optimizer LLM. At every round the
optimizer LLM chooses how to revise the current skill: a search strategy (I1 direct revision, I2 iterative
refinement, I3 parallel sampling) and the revision forms applied under it (F1 rules, F2 examples, F3
reasoning procedure, F4 full rewrite). Every candidate skill passes the same candidate evaluation, initial
screening and then further validation, before it can replace the current skill. This repository contains
the code, the configurations and learned skills of the paper's runs, and a small sample of each benchmark.

## Installation

Python 3.11 or later.

```bash
git clone https://github.com/seai-lab/StraTune.git && cd StraTune
pip install -r requirements.txt
```

Models are called through Amazon Bedrock. Configure AWS credentials with Bedrock access (`AWS_PROFILE`,
access keys, or `AWS_BEARER_TOKEN_BEDROCK`) and set `AWS_REGION` (the paper used `us-west-2`).

| Setting | Target LLM | Optimizer LLM |
|---|---|---|
| Haiku/Sonnet (main) | `global.anthropic.claude-haiku-4-5-20251001-v1:0` | `global.anthropic.claude-sonnet-4-6` |
| Opus | `us.anthropic.claude-opus-4-8` | `us.anthropic.claude-opus-5` |

## Quick start

`data/benchmark_sample/` holds 40 training and 20 test tasks of each benchmark, enough to run every
stage of the code; the numbers it produces are not the paper's.

```bash
export AWS_REGION=us-west-2
export STRATUNE_BENCHMARK_DATA=$PWD/data/benchmark_sample
python3 -B method/tests/smoke_test.py                              # offline checks
python3 -m method.environments.datasets.materialize_docvqa train   # DocVQA images, a few seconds
python3 -m method.environments.datasets.materialize_docvqa test
method/scripts/train.sh livemath haiku --dry-run                   # 4 short rounds
method/scripts/train.sh livemath                                   # full loop on the sample
method/scripts/evaluate.sh --skill data/skills/haiku/livemath.txt --dataset livemath
```

## Full benchmark data

The exact splits of the paper (about 1 GB, mostly DocVQA page images) are on the Hugging Face Hub:

```bash
hf download PingL/StraTune_dataset --repo-type dataset --local-dir benchmark_data
for s in train test; do tar -xzf benchmark_data/SpreadsheetBench/$s/spreadsheet.tar.gz -C benchmark_data/SpreadsheetBench/$s; done
export STRATUNE_BENCHMARK_DATA=$PWD/benchmark_data
python3 -m method.environments.datasets.materialize_docvqa train
python3 -m method.environments.datasets.materialize_docvqa test
```

| Dataset | Train | Test | Metric |
|---|---:|---:|---|
| DocVQA | 500 | 1,000 | ANLS |
| LiveMathematicianBench | 397 | 211 | answer accuracy |
| Mind2Web | 800 | 252 | element accuracy |
| SpreadsheetBench | 629 | 280 | fraction of tasks passing all checks |

## Training and evaluation

```bash
method/scripts/train.sh <docvqa|livemath|mind2web|spreadsheetbench> [haiku|opus] [--dry-run]
method/scripts/evaluate.sh --run-id <run_id> --workers 8
method/scripts/evaluate.sh --skill data/skills/haiku/livemath.txt --dataset livemath --workers 8
```

`train.sh` selects the hyperparameters in `method/configs/` and the model ids of the setting, and runs
with a training budget of 6 |D_train| target-LLM executions (3000 / 2382 / 4800 / 3774 for DocVQA /
LiveMath / Mind2Web / SpreadsheetBench). Runs are written to `runtime_data/runs/<run_id>/`, with the
final skill in `final_skill.txt`. `evaluate.sh` scores a completed run or any skill file, for example the
released skills in `data/skills/`, on the test split; set `STRATUNE_TARGET_MODEL` to the target LLM of
the setting. Both LLMs run at temperature 0; results still vary across runs because of API
nondeterminism and the samples drawn in each round. Model ids and paths can be overridden with
`STRATUNE_TARGET_MODEL`, `STRATUNE_OPTIMIZER_MODEL`, `STRATUNE_HP_JSON`, `STRATUNE_BENCHMARK_DATA`, and
`STRATUNE_ROOT`. The baselines of the paper were run with their public implementations and are not part
of this repository.

## Repository structure

```text
method/
  train.py, evaluate.py, config.py   training loop, test evaluation, datasets and budgets
  operators/                         revision operators: strategy and form selection, I1-I3, F4
  evaluation/                        candidate evaluation: screening sets, paired execution, acceptance rules
  state/                             optimization state: current skill, execution feedback, histories
  environments/                      task environments, dataset loaders, metrics, Bedrock client
  configs/, scripts/, tests/         hyperparameters of the paper's runs, train.sh / evaluate.sh, smoke test
data/
  benchmark_sample/                  40 train + 20 test tasks per benchmark for running the code
  strata/                            per-task stratification fields
  initial_skills/, skills/           initial skill of each dataset; final skills of the runs in Table 1
```

## Citation

```bibtex
@article{liu2026stratune,
  title   = {StraTune: Adaptive Selection of Revision Operators for Self-Evolving LLM Skills},
  author  = {Liu, Zeping and Li, Yan and Lao, Ni and Wolff, Gil and Mai, Gengchen},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

The code is released under the MIT License. The benchmark data keep the licenses of their sources,
listed on the [dataset page](https://huggingface.co/datasets/PingL/StraTune_dataset).
