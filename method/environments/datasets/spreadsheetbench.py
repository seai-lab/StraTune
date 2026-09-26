"""SpreadsheetBench loader, reading ${STRATUNE_BENCHMARK_DATA}/SpreadsheetBench/{train,test}/:
tasks.json lists the tasks with their case files, which live under spreadsheet/<task>/.

    train -> 629 tasks from all_data_912_v0.1 (three workbook cases each, a few with fewer)
    test  -> 280 tasks, the SkillOpt official test split over spreadsheetbench_verified_400

Dependencies: stdlib (+ openpyxl only when .open_workbook() is used).
"""
from .common import BENCHMARK_DATA, check_tier, load_json

SPREADSHEETBENCH_DIR = BENCHMARK_DATA / "SpreadsheetBench"


class SpreadsheetBenchCase:
    __slots__ = ("case", "input_path", "answer_path", "naming")

    def __init__(self, case, input_path, answer_path, naming=None):
        self.case = case
        self.input_path = input_path    # absolute Path to the case input workbook
        self.answer_path = answer_path  # absolute Path to the expected workbook
        self.naming = naming            # "standard"/"initial_golden"/"fuzzy_prefixed" (test tier)

    def open_workbook(self, which="input", data_only=True):
        import openpyxl

        path = {"input": self.input_path, "answer": self.answer_path}[which]
        return openpyxl.load_workbook(path, data_only=data_only)

    def __repr__(self):
        return f"SpreadsheetBenchCase(case={self.case}, input={self.input_path.name!r})"


class SpreadsheetBenchTask:
    __slots__ = ("task_id", "instruction", "instruction_type", "answer_position",
                 "answer_sheet", "data_position", "dir_path", "cases",
                 "prompt_file", "payload", "tier")

    def __init__(self, entry, payload_root, tier):
        self.task_id = entry["id"]
        self.tier = tier
        self.payload = entry["payload"]
        self.instruction = entry.get("instruction", "")
        self.instruction_type = entry.get("instruction_type")
        self.answer_position = entry.get("answer_position")
        self.answer_sheet = entry.get("answer_sheet")
        self.data_position = entry.get("data_position")
        self.dir_path = payload_root / entry["dir"]
        self.cases = [
            SpreadsheetBenchCase(
                case=c["case"],
                input_path=self.dir_path / c["input"],
                answer_path=self.dir_path / c.get("answer", c.get("golden")),
                naming=c.get("naming"),
            )
            for c in entry["cases"]
        ]
        pf = entry.get("prompt_file")
        self.prompt_file = (self.dir_path / pf) if pf else None

    @property
    def n_cases(self):
        return len(self.cases)

    def __repr__(self):
        return (f"SpreadsheetBenchTask(id={self.task_id!r}, tier={self.tier!r}, "
                f"n_cases={self.n_cases})")


class SpreadsheetBenchDataset:
    def __init__(self, root=None):
        self.root = root or SPREADSHEETBENCH_DIR

    def task_entries(self, tier):
        check_tier(tier)
        return load_json(self.root / tier / "tasks.json")

    def task_ids(self, tier):
        return [str(e["id"]) for e in self.task_entries(tier)]

    def iter_tasks(self, tier, verify_files=True):
        """Yield SpreadsheetBenchTask in tasks.json order. With verify_files
        (default) every referenced case file must exist on disk."""
        check_tier(tier)
        for e in self.task_entries(tier):
            task = SpreadsheetBenchTask(e, self.root / tier, tier)
            if verify_files:
                for c in task.cases:
                    if not c.input_path.is_file():
                        raise FileNotFoundError(f"{task.task_id}: missing {c.input_path}")
                    if not c.answer_path.is_file():
                        raise FileNotFoundError(f"{task.task_id}: missing {c.answer_path}")
            yield task
