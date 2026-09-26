# StraTune: Adaptive Selection of Revision Operators for Self-Evolving LLM Skills

Code, configurations, and learned skills for the paper *StraTune: Adaptive Selection of Revision
Operators for Self-Evolving LLM Skills* (arXiv:XXXX.XXXXX). The benchmark data are at
https://huggingface.co/datasets/PingL/StraTune_dataset. StraTune
optimizes a textual skill for a frozen target LLM with a frozen optimizer LLM. At every round the
optimizer LLM chooses a revision operator, a search strategy (I1 direct revision, I2 iterative
refinement, I3 parallel sampling) together with the revision forms applied under it (F1 conditional
rules, F2 worked examples, F3 reasoning procedure, F4 full rewrite), from the optimization state, and
every candidate skill passes one candidate evaluation (initial screening, further validation) before it
can replace the current skill.

## Layout

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

## Setup

Python 3.11 or later.

```bash
pip install -r requirements.txt
```

Models are called through Amazon Bedrock (Converse API). Configure AWS credentials with Bedrock
access in the usual way (`AWS_PROFILE`, access keys, or `AWS_BEARER_TOKEN_BEDROCK`) and set
`AWS_REGION` (the paper used `us-west-2`). Model ids default to the paper's setting and can be
overridden with `STRATUNE_TARGET_MODEL` and `STRATUNE_OPTIMIZER_MODEL`:

| Setting | Target LLM | Optimizer LLM |
|---|---|---|
| Haiku/Sonnet (main) | `global.anthropic.claude-haiku-4-5-20251001-v1:0` | `global.anthropic.claude-sonnet-4-6` |
| Opus | `us.anthropic.claude-opus-4-8` | `us.anthropic.claude-opus-5` |

## Quick start on the sample data

`data/benchmark_sample/` holds 40 training and 20 test tasks of each benchmark in the layout of
the full `benchmark_data/` (about 5 MB). It is enough to run every stage of the code (strategy selection, I1/I2/I3 generation, screening,
final skill selection); the set sizes and budget margins are reduced automatically for it, and the
numbers it produces are not the paper's.

```bash
pip install -r requirements.txt
export AWS_REGION=us-west-2                       # plus Bedrock credentials
export STRATUNE_BENCHMARK_DATA=$PWD/data/benchmark_sample
python3 -B method/tests/smoke_test.py            # offline checks
python3 -m method.environments.datasets.materialize_docvqa train   # DocVQA only, a few seconds
python3 -m method.environments.datasets.materialize_docvqa test
method/scripts/train.sh livemath haiku --dry-run  # 4 short rounds, calls the models
method/scripts/train.sh livemath                  # full loop on the 40 training tasks (budget 6 x 40)
method/scripts/evaluate.sh --skill data/skills/haiku/livemath.txt --dataset livemath
```

## Data

This repository includes `data/benchmark_sample/`, a small subset of the four benchmarks on which all
commands in this README run. The full benchmark payloads (about 1 GB, mostly DocVQA document
images) are needed to reproduce the numbers of the paper and are hosted on the Hugging Face Hub:

```bash
pip install -U huggingface_hub
hf download PingL/StraTune_dataset --repo-type dataset --local-dir benchmark_data
for s in train test; do tar -xzf benchmark_data/SpreadsheetBench/$s/spreadsheet.tar.gz -C benchmark_data/SpreadsheetBench/$s; done
export STRATUNE_BENCHMARK_DATA=$PWD/benchmark_data
```

The second line unpacks the SpreadsheetBench workbooks, which are stored as one archive per split.
The full data have the same layout as the sample, and `STRATUNE_BENCHMARK_DATA` selects which one is used. Each dataset has `train/` and `test/` subdirectories whose `task_ids.json` define
the exact splits of the paper:

```text
benchmark_data/
├── DocVQA/dev/{train,test}/                   # one parquet file with the document images + task_index.json
├── LiveMathematicianBench/{train,test}/       # tasks.jsonl, labels.jsonl
├── Mind2Web/train/tasks.jsonl                 # official train shards, 20 candidate elements per step
├── Mind2Web/test/tasks.json                   # official test_task split, prepared the same way
└── SpreadsheetBench/{train,test}/             # tasks.json + spreadsheet/<task>/ workbooks
```

| Dataset | Train | Test | Metric |
|---|---:|---:|---|
| DocVQA | 500 | 1,000 | ANLS |
| LiveMathematicianBench | 397 | 211 | answer accuracy |
| Mind2Web | 800 | 252 | element accuracy |
| SpreadsheetBench | 629 | 280 | fraction of tasks passing all checks |

DocVQA rollouts read a materialized copy of each split (task records plus PNG images), written to
`runtime_data/docvqa/` in a few seconds:

```bash
export STRATUNE_BENCHMARK_DATA=/path/to/benchmark_data
python3 -m method.environments.datasets.materialize_docvqa train
python3 -m method.environments.datasets.materialize_docvqa test
```

## Training

```bash
method/scripts/train.sh livemath                 # Haiku/Sonnet setting
method/scripts/train.sh livemath opus            # Opus setting
method/scripts/train.sh livemath haiku --dry-run # shortened run (4 rounds); still calls the models
```

The script selects the hyperparameter file (`method/configs/<dataset>.json`; `livemath_opus.json`
for LiveMath with Opus) and the model ids, then runs `python3 -m method.train <dataset>`. Every run
has a training budget of `6 |D_train|` target-LLM executions (3000 / 2382 / 4800 / 3774 for
DocVQA / LiveMath / Mind2Web / SpreadsheetBench), covering execution feedback, candidate evaluation,
and final skill selection. Run state, skills, and logs are written to `runtime_data/runs/<run_id>/`;
the final skill is `runtime_data/runs/<run_id>/final_skill.txt`. Both LLMs are called at temperature 0;
results still vary across runs because of API nondeterminism and the samples drawn in each round.

## Evaluation

```bash
method/scripts/evaluate.sh --run-id <run_id> --workers 8
method/scripts/evaluate.sh --skill data/skills/haiku/livemath.txt --dataset livemath --workers 8
```

With `--run-id`, `method/evaluate.py` resolves the final skill from the completed run's manifest and
evaluates it on the official test split; results are written to the run directory. With `--skill`, any
skill file is evaluated, for example the released skills of Table 1 in `data/skills/`; results are
written to `runtime_data/skill_eval/`. Set `STRATUNE_TARGET_MODEL` to the target LLM of the setting
(the Opus skills were learned and evaluated with the Opus target LLM).

## Baselines

The baselines (TextGrad, GEPA, SkillOpt, Trace2Skill, Iter-CoT) were run with their public
implementations on the same environments, initial skills, model settings, test splits, and training
budget; their code is not part of this release.

## Tests

```bash
python3 -B method/tests/smoke_test.py
```

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `STRATUNE_ROOT` | repository root | directory of this README |
| `STRATUNE_BENCHMARK_DATA` | benchmark payload root | `<root>/benchmark_data` |
| `STRATUNE_HP_JSON` | hyperparameter file | `method/configs/<dataset>.json` |
| `STRATUNE_TARGET_MODEL`, `STRATUNE_OPTIMIZER_MODEL` | Bedrock model ids | Haiku 4.5 / Sonnet 4.6 |
| `STRATUNE_EXPECTED_ROLE` | optional substring the AWS caller ARN must contain | unset |

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

The code is released under the MIT License (see `LICENSE`). The benchmark samples in `data/benchmark_sample/`
and the full data on the Hugging Face Hub remain under the licenses of their sources, listed on the
dataset page.
