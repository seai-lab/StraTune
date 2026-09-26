#!/usr/bin/env python3
"""Materialize a DocVQA split for thread-safe parallel rollouts.

The parquet image cache is not thread-safe, so this script does one sequential
pass over the split's parquet file and writes the tasks to:

    runtime_data/docvqa/<tier><n>/
    ├── tasks.jsonl   (qid, question, question_types, answers, doc_id, ...)
    ├── images/<qid>.png
    └── MANIFEST.json (ids_sha256 of the split, counts, aggregate image sha256)

Usage: python3 -m method.environments.datasets.materialize_docvqa <train|test>
"""
import hashlib
import json
import os
import sys

_STRATUNE_ROOT = os.environ.get("STRATUNE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
WORKDIR = _STRATUNE_ROOT
sys.path.insert(0, WORKDIR)
os.environ.setdefault("STRATUNE_WORKSPACE", WORKDIR)

from method.common import util  # noqa: E402


def materialized_dir(subset: dict) -> str:
    return os.path.join(util.DATA_ROOT, "docvqa", f"{subset['split']}{subset['n']}")


def main(tier: str):
    from method.environments.datasets.docvqa import DocVQADataset
    from method.environments.datasets.splits import load_split

    subset = load_split("docvqa", tier)
    out_dir = materialized_dir(subset)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    manifest_file = os.path.join(out_dir, "MANIFEST.json")
    if os.path.exists(manifest_file):
        m = util.read_json(manifest_file)
        if m["ids_sha256"] == subset["ids_sha256"] and m["complete"]:
            print(f"already materialized: {out_dir}")
            return

    want = set(subset["ids"])
    found = 0
    agg = hashlib.sha256()
    with open(os.path.join(out_dir, "tasks.jsonl.tmp"), "w") as tf:
        for task in DocVQADataset().iter_tasks(tier):
            qid = str(task.qid)
            if qid not in want:
                continue
            png = bytes(task.image_bytes())
            with open(os.path.join(img_dir, f"{qid}.png"), "wb") as f:
                f.write(png)
            rec = {
                "qid": qid,
                "question": task.question,
                "question_types": list(task.question_types or []),
                "answers": list(task.answers or []),
                "doc_id": task.doc_id,
                "png_bytes": len(png),
                "png_sha256": hashlib.sha256(png).hexdigest(),
            }
            tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            agg.update(rec["png_sha256"].encode())
            found += 1
            if found % 512 == 0:
                print(f"  {found}/{len(want)}", flush=True)
    assert found == len(want), f"found {found} of {len(want)}"
    os.replace(os.path.join(out_dir, "tasks.jsonl.tmp"),
               os.path.join(out_dir, "tasks.jsonl"))
    util.atomic_write_json(manifest_file, {
        "ids_sha256": subset["ids_sha256"],
        "n": found,
        "tier": tier,
        "images_agg_sha256": agg.hexdigest(),
        "complete": True,
    })
    print(f"materialized {found} tasks -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
