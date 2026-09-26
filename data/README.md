# data/

Everything a training or evaluation run reads besides the benchmark payloads
(`benchmark_data/`, released separately; its `train/task_ids.json` and
`test/task_ids.json` define the splits).

| Directory | Contents | Read by |
|---|---|---|
| `benchmark_sample/` | 40 training and 20 test tasks of each benchmark in the layout of `benchmark_data/` (DocVQA and SpreadsheetBench: the tasks with the smallest payloads; LiveMath and Mind2Web: the first ids). `SAMPLE.json` marks it as a sample so the split-size checks are skipped. For running the code only; not the paper's splits. | all loaders, when `STRATUNE_BENCHMARK_DATA` points here |
| `strata/` | Stratum of every training task, one JSON per dataset mapping task id to its fields. The fields used (`STRATA_FIELDS` in `method/config.py`) are question length for DocVQA, release month for LiveMath, website and step count for Mind2Web, and instruction type and length for SpreadsheetBench. Used to form the 16 stratified training batches and the stratified random part of each screening set. | `method.environments.datasets.splits` (batches), `method.evaluation.screening_sets` |
| `initial_skills/` | The initial skill $s_0$ of each dataset, the same for every method in the paper. | `method.train` |
| `skills/` | The final skills of the StraTune runs reported in Table 1 (`haiku/` and `opus/`); see `skills/README.md`. | `method.evaluate --skill` |
