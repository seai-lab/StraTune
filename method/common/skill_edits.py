"""Skill representation and mechanical patch application.

A skill is plain text (the system prompt). A patch is a dict:
    {update_id, parent_version, operation: replace|append|insert_after,
     old_content, new_content, scope: [str], rationale,
     source_task_ids, source_batch_id}
Application is mechanical and reversible; everything outside the edited span
is left unchanged.
"""
from __future__ import annotations


class PatchError(ValueError):
    """Structurally invalid, non-applicable, or oversized full-rewrite patch."""


# Length constants in approximate tokens. Local edits have no token-length
# limit; EDIT_CAP bounds the content merged from saved candidate skills, and a
# full rewrite is capped at max(REWRITE_MULT * parent, parent + REWRITE_SLACK).
EDIT_CAP = 1200
EDIT_FLOOR = 300
GROWTH_FRAC = 0.40
REWRITE_MULT = 3       # rewrite cap = max(MULT*parent, parent+REWRITE_SLACK)
REWRITE_SLACK = 1500


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def check_edit_budget(skill: str, patch: dict):
    """Local edits have no token-length limit; always accepts."""
    return None


def validate_patch(skill: str, patch: dict):
    op = patch.get("operation")
    old = patch.get("old_content") or ""
    new = patch.get("new_content") or ""
    if op not in ("replace", "append", "insert_after"):
        raise PatchError(f"unknown operation {op!r}")
    if op == "replace":
        if not old:
            raise PatchError("replace requires old_content")
        n = skill.count(old)
        if n != 1:
            raise PatchError(f"old_content occurs {n} times (need exactly 1)")
        if not new:
            raise PatchError("replace with empty new_content (use a real edit)")
    elif op == "insert_after":
        if not old:
            raise PatchError("insert_after requires an anchor old_content")
        if skill.count(old) != 1:
            raise PatchError("insert_after anchor must occur exactly once")
        if not new:
            raise PatchError("insert_after requires new_content")
    else:  # append
        if not new:
            raise PatchError("append requires new_content")
    check_edit_budget(skill, patch)


def apply_rewrite(skill: str, patch: dict) -> str:
    """Full-rewrite patch: new_content replaces the whole skill.
    Size cap: <= max(REWRITE_MULT x parent, parent + SLACK)."""
    new = (patch.get("new_content") or "").strip()
    if not new:
        raise PatchError("rewrite requires non-empty new_content")
    cap = max(REWRITE_MULT * approx_tokens(skill),
              approx_tokens(skill) + REWRITE_SLACK)
    if approx_tokens(new) > cap:
        raise PatchError(f"rewrite too large: {approx_tokens(new)} > {cap} tokens")
    return new + ("\n" if not new.endswith("\n") else "")


def apply_patch(skill: str, patch: dict) -> str:
    """Apply a single-edit patch, a multi-edit patch {edits: [...]}, or a
    full rewrite {operation: rewrite}."""
    if patch.get("operation") == "rewrite":
        return apply_rewrite(skill, patch)
    if "edits" in patch:
        return apply_multi(skill, patch)
    validate_patch(skill, patch)
    op = patch["operation"]
    old = patch.get("old_content") or ""
    new = patch.get("new_content") or ""
    if op == "replace":
        return skill.replace(old, new, 1)
    if op == "insert_after":
        idx = skill.index(old) + len(old)
        return skill[:idx] + "\n\n" + new.strip("\n") + skill[idx:]
    return skill.rstrip("\n") + "\n\n" + new.strip("\n") + "\n"


def _apply_one(skill: str, edit: dict) -> str:
    op = edit["operation"]
    old = edit.get("old_content") or ""
    new = edit.get("new_content") or ""
    if op == "replace":
        return skill.replace(old, new, 1)
    if op == "insert_after":
        idx = skill.index(old) + len(old)
        return skill[:idx] + "\n\n" + new.strip("\n") + skill[idx:]
    return skill.rstrip("\n") + "\n\n" + new.strip("\n") + "\n"


def apply_multi(skill: str, patch: dict) -> str:
    """Apply 1-6 local edits in order, without a token-length limit."""
    edits = patch.get("edits") or []
    if not 1 <= len(edits) <= 6:
        raise PatchError(f"multi-edit patch needs 1-6 edits, got {len(edits)}")
    cur = skill
    for i, e in enumerate(edits):
        single = dict(e)
        try:
            # validate structure against the CURRENT intermediate text
            op = single.get("operation")
            old = single.get("old_content") or ""
            new = single.get("new_content") or ""
            if op not in ("replace", "append", "insert_after"):
                raise PatchError(f"edit {i}: unknown operation {op!r}")
            if op in ("replace", "insert_after"):
                if not old:
                    raise PatchError(f"edit {i}: {op} requires old_content")
                if cur.count(old) != 1:
                    raise PatchError(f"edit {i}: anchor occurs {cur.count(old)} times")
            if not new and op != "replace":
                raise PatchError(f"edit {i}: requires new_content")
            cur = _apply_one(cur, single)
        except PatchError:
            raise
    return cur


def _flatten_edits(patch: dict) -> list:
    """A patch is either single-edit (operation/old/new) or multi-edit
    ({edits: [...]}) — normalize to an edit list."""
    if "edits" in patch:
        return list(patch["edits"])
    return [{k: patch.get(k) for k in ("operation", "old_content", "new_content")}]


def canonical_compose(base_skill: str, patches: list) -> str:
    """Canonical composition: flatten all patches' edits, order by anchor
    position in the base skill (replace/insert_after by base.find(old);
    appends last, stable in the given patch order), apply sequentially. Raises
    PatchError if any anchor is missing or ambiguous when applied; such a
    set of patches is not mergeable."""
    keyed = []
    for pi, patch in enumerate(patches):
        for ei, e in enumerate(_flatten_edits(patch)):
            op = e.get("operation")
            old = e.get("old_content") or ""
            if op in ("replace", "insert_after"):
                pos = base_skill.find(old) if old else -1
                if pos < 0:
                    raise PatchError(f"anchor of patch#{pi} edit#{ei} not in base")
            else:
                pos = 1 << 30  # appends last
            keyed.append((pos, pi, ei, e))
    keyed.sort(key=lambda x: (x[0], x[1], x[2]))
    cur = base_skill
    for pos, pi, ei, e in keyed:
        op = e.get("operation")
        old = e.get("old_content") or ""
        new = e.get("new_content") or ""
        if op in ("replace", "insert_after"):
            if cur.count(old) != 1:
                raise PatchError(
                    f"anchor of patch#{pi} edit#{ei} occurs {cur.count(old)}x after prior edits")
        elif op != "append":
            raise PatchError(f"unknown operation {op!r}")
        if not new and op != "replace":
            raise PatchError(f"patch#{pi} edit#{ei} missing new_content")
        cur = _apply_one(cur, e)
    return cur


def patches_mergeable(base_skill: str, patch_a: dict, patch_b: dict) -> bool:
    try:
        canonical_compose(base_skill, [patch_a, patch_b])
        return True
    except PatchError:
        return False


def patches_compatible(skill_parent: str, patch_a: dict, patch_b: dict) -> bool:
    """Structural compatibility of two patches: both apply to the common
    parent individually and sequentially in either order (the edited regions
    do not collide)."""
    try:
        ka = apply_patch(skill_parent, patch_a)
        kb = apply_patch(skill_parent, patch_b)
        kab = apply_patch(ka, patch_b)
        kba = apply_patch(kb, patch_a)
    except PatchError:
        return False
    return kab == kba


def merge_patches(skill_parent: str, patch_a: dict, patch_b: dict) -> str:
    if not patches_compatible(skill_parent, patch_a, patch_b):
        raise PatchError("patches are not structurally compatible")
    return apply_patch(apply_patch(skill_parent, patch_a), patch_b)
