"""Benchmark loaders (DocVQA, Mind2Web, SpreadsheetBench) and the paper's splits.

All loaders read only the fixed benchmark payloads under
``${STRATUNE_BENCHMARK_DATA}``. No network, no LLM calls. Tier names are
``"train"`` and ``"test"`` everywhere.
"""

from .docvqa import DocVQADataset
from .mind2web import Mind2WebDataset
from .splits import load_split
from .spreadsheetbench import SpreadsheetBenchDataset

__all__ = ["DocVQADataset", "Mind2WebDataset", "SpreadsheetBenchDataset", "load_split"]
