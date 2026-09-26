"""Optimizer-LLM call helpers, the content requirements shared by the generation
prompts, and the prompt that merges content from a saved candidate skill into the
current skill before final skill selection.

Calls use temperature 0.0 and JSON output with one repair retry. The optimizer
LLM never sees per-sample screening results, only the execution feedback and the
evaluation history.
"""
import json
import re

from method.config import OPTIMIZER_MODEL
from method.common.execution import client_for


class LLMOpError(RuntimeError):
    pass


def _call(prompt: str, tag: str, system: str | None = None) -> str:
    client = client_for("optimizer/stratune")
    resp = client.converse(
        OPTIMIZER_MODEL, [{"role": "user", "content": [{"text": prompt}]}],
        system=system, max_tokens=16384, temperature=0.0, seed=42,
        tags={"stage": tag})
    return resp["text"]


def _extract_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)


def _call_json(prompt: str, tag: str) -> dict:
    text = _call(prompt, tag)
    try:
        return _extract_json(text)
    except (json.JSONDecodeError, ValueError):
        repair = (f"Your previous reply was not parseable JSON. Reply with ONLY the "
                  f"corrected JSON object, no prose.\n\nPrevious reply:\n{text[:6000]}")
        text2 = _call(repair, tag + "_repair")
        try:
            return _extract_json(text2)
        except (json.JSONDecodeError, ValueError) as e:
            raise LLMOpError(f"unparseable optimizer JSON after repair ({tag})") from e


# ---------------------------------------------------------------------------
# Content style requirements appended to the revision and rewrite prompts when
# CONTENT_V25 is enabled (the training driver enables it).
CONTENT_V25 = False

V25_CONTENT_SPEC = """
CONTENT STYLE REQUIREMENTS (these forms measurably outperform abstract advice):
- CONTRAST PAIRS: express format/extraction rules as concrete wrong->right \
pairs, e.g. `WRONG: "The date is March 22, 1976." -> RIGHT: "March 22, 1976"`. \
Derive them from the verified fixes; generalize names/values so no task answer \
is memorized. Prefer 2-4 such pairs over a paragraph of prose.
- ANSWER SHAPE: state exactly what the expected answer STRING looks like for \
each question kind (bare value vs sentence, category word kept or dropped, \
units, capitalization, verbatim copying), with a one-line reason.
- TRIGGER -> ACTION: write rules as "When <observable condition in the input \
or your draft answer>, do <specific action>" — not "be careful about X".
- ESCALATION: if the evidence shows one dominant recurring mistake, do not \
merely mention the correction — make it a MANDATORY first-step check or a \
short veto list ("if your reasoning contains phrases like <...>, stop and \
reconsider toward <default>"), and state the default direction explicitly.
- INPUT QUIRKS: if the failures reveal dataset artifacts (truncated options, \
separator characters with two meanings, OCR character confusions), add a \
short handling protocol for the artifact itself.
"""


SEMANTIC_MERGE_PROMPT = """You maintain the system prompt (the "skill") of an AI agent. During training, a
CANDIDATE variant of this skill was NOT adopted overall, yet the paired ledger shows it
SOLVED specific tasks the current champion fails (listed below). Integrate the candidate's
plausibly-responsible content into the champion as a small set of surgical edits.

CHAMPION SKILL (the base you are editing):
<skill>
{skill}
</skill>

DONOR CANDIDATE (not adopted; mine it for what the champion lacks):
<donor>
{donor}
</donor>

TASKS THE DONOR RESCUES (champion fails these; the donor's differing content likely explains why):
{rescues}

{postmortem}

Edit rules:
- 1-4 edits, operations replace/insert_after/append; old_content must be an EXACT UNIQUE
  substring of the champion skill. NEVER rewrite the whole skill.
- Integrate ONLY content that plausibly explains the rescues; keep every champion rule that
  is not contradicted; keep exact output-format instructions VERBATIM.
- DEDUPLICATE while you are here: if the champion contains near-duplicate paragraphs or a
  rule stated twice, one edit may consolidate them.
- General rules/protocols only — never task-specific answers.
- Return complete edits without a separate token-length limit.

Reply with ONLY JSON:
{{"edits": [{{"operation": "replace|insert_after|append",
             "old_content": "<exact substring or empty>",
             "new_content": "<text>"}}, ...],
  "scope": ["<keywords>"], "rationale": "<1-2 sentences>"}}"""


def propose_semantic_merge(dataset: str, skill: str, donor: str, rescues: list[str],
                           postmortem: str, max_tokens: int, salt: str = "") -> dict:
    pm = (f"WHY THE DONOR WAS REJECTED OVERALL (avoid re-importing this failure):\n{postmortem}"
          if postmortem else "")
    out = _call_json(SEMANTIC_MERGE_PROMPT.format(
        skill=skill, donor=donor, rescues="\n".join(rescues), postmortem=pm,
        max_tokens=max_tokens) + (f"\n\n<!-- {salt} -->" if salt else ""),
        tag="semantic_merge")
    if not isinstance(out.get("edits"), list) or not out["edits"]:
        raise LLMOpError("semantic merge missing edits")
    out.setdefault("scope", ["semantic merge"])
    out.setdefault("rationale", "")
    out["operation"] = "multi"
    out["new_content"] = " ".join(e.get("new_content") or "" for e in out["edits"])
    return out

