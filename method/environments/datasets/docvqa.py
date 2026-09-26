"""DocVQA loader: the 500 training and 1,000 test questions of the paper, read
from ${STRATUNE_BENCHMARK_DATA}/DocVQA/dev/{train,test}/ (one parquet file with
the document images, plus task_index.json).

Dependencies: stdlib + pyarrow (+ pillow only when .image() is called).
"""
from .common import BENCHMARK_DATA, check_tier, load_json

DOCVQA_DIR = BENCHMARK_DATA / "DocVQA" / "dev"


class _RowGroupImageCache:
    """Caches the image column of the most recently read row group."""

    def __init__(self):
        self._key = None
        self._bytes_list = None

    def image_bytes(self, shard_path, row_group, row_in_group):
        import pyarrow.parquet as pq

        key = (str(shard_path), row_group)
        if key != self._key:
            pf = pq.ParquetFile(shard_path)
            tbl = pf.read_row_group(row_group, columns=["image"])
            col = tbl.column("image").combine_chunks()
            self._bytes_list = col.field("bytes").to_pylist()
            self._key = key
        return self._bytes_list[row_in_group]


class DocVQATask:
    __slots__ = ("qid", "question", "question_types", "answers", "doc_id",
                 "ucsf_document_id", "ucsf_document_page_no",
                 "_shard_path", "_row_group", "_row_in_group", "_cache")

    def __init__(self, qid, question, question_types, answers, doc_id,
                 ucsf_document_id, ucsf_document_page_no,
                 shard_path, row_group, row_in_group, cache):
        self.qid = qid
        self.question = question
        self.question_types = question_types
        self.answers = answers
        self.doc_id = doc_id
        self.ucsf_document_id = ucsf_document_id
        self.ucsf_document_page_no = ucsf_document_page_no
        self._shard_path = shard_path
        self._row_group = row_group
        self._row_in_group = row_in_group
        self._cache = cache

    def image_bytes(self):
        """Raw encoded image bytes (PNG), read from parquet on demand."""
        return self._cache.image_bytes(self._shard_path, self._row_group,
                                       self._row_in_group)

    def image(self):
        """PIL.Image, decoded on demand."""
        import io

        from PIL import Image

        return Image.open(io.BytesIO(self.image_bytes()))

    def __repr__(self):
        return f"DocVQATask(qid={self.qid!r}, doc_id={self.doc_id})"


class DocVQADataset:
    def __init__(self, root=None):
        self.root = root or DOCVQA_DIR
        self._index_cache = {}

    def task_entries(self, tier):
        """task_index.json entries: [{qid, docId, file, row, n_answers, ...}, ...] in file order."""
        check_tier(tier)
        if tier not in self._index_cache:
            self._index_cache[tier] = load_json(self.root / tier / "task_index.json")
        return self._index_cache[tier]

    def task_ids(self, tier):
        return [str(e["qid"]) for e in self.task_entries(tier)]

    def iter_tasks(self, tier):
        """Yield DocVQATask for every entry of `tier`, in file order, checking the
        questionId of each row. Images are read on demand."""
        import pyarrow.parquet as pq

        entries = self.task_entries(tier)
        cache = _RowGroupImageCache()
        columns = ["questionId", "question", "question_types", "answers",
                   "docId", "ucsf_document_id", "ucsf_document_page_no"]
        pos = 0
        for fname in dict.fromkeys(e["file"] for e in entries):
            shard_path = self.root / tier / fname
            pf = pq.ParquetFile(shard_path)
            row = 0
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=columns)
                cols = {c: tbl.column(c).to_pylist() for c in columns}
                for i in range(tbl.num_rows):
                    entry = entries[pos]
                    if entry["file"] != fname or entry["row"] != row:
                        raise AssertionError(f"task_index misalignment at {fname} row {row}")
                    if str(cols["questionId"][i]) != str(entry["qid"]):
                        raise AssertionError(
                            f"questionId mismatch at {fname} row {row}: "
                            f"{cols['questionId'][i]!r} != {entry['qid']!r}")
                    yield DocVQATask(
                        qid=str(cols["questionId"][i]),
                        question=cols["question"][i],
                        question_types=cols["question_types"][i],
                        answers=cols["answers"][i],
                        doc_id=cols["docId"][i],
                        ucsf_document_id=cols["ucsf_document_id"][i],
                        ucsf_document_page_no=cols["ucsf_document_page_no"][i],
                        shard_path=shard_path, row_group=rg, row_in_group=i, cache=cache)
                    pos += 1
                    row += 1
        if pos != len(entries):
            raise AssertionError(f"yielded {pos} tasks, task_index has {len(entries)}")
