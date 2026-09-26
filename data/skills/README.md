# Final skills of the StraTune runs in Table 1

Each file is the final skill produced by one StraTune training run and evaluated on the test split of its dataset.

| Directory | Target LLM | Optimizer LLM | Table 1 column |
|---|---|---|---|
| `haiku/` | Claude Haiku 4.5 | Claude Sonnet 4.6 | Haiku/Sonnet |
| `opus/` | Claude Opus 4.8 | Claude Opus 5 | Opus |

Files: `docvqa.txt`, `livemath.txt`, `mind2web.txt`, `spreadsheetbench.txt`. The corresponding initial skills are in `data/initial_skills/`. To evaluate a skill on the test split, run for example `method/scripts/evaluate.sh --skill data/skills/haiku/livemath.txt --dataset livemath` with `STRATUNE_TARGET_MODEL` set to the target LLM of that setting.
