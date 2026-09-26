"""Acceptance decision of initial screening.

Input: paired statistics of a candidate skill against the current skill, from
one screening set or from two disjoint screening sets combined (counts summed,
gains averaged over the union). Output: one of commit (accept), branch (save
without acceptance), reject, probation (inconclusive; screen again on a disjoint
set), or insufficient_evidence, together with the reasons.

Regression proportions are compared through one-sided 90% Wilson lower bounds
rather than point estimates, so that a few losses on a small set do not by
themselves reject a candidate.
"""
import math


_Z90 = 1.2816  # one-sided 90%

# When set by method.train, a candidate whose combined screening evidence is still
# inconclusive but whose gain is positive and whose regression checks pass is
# saved for final skill selection instead of being discarded.
ARCHIVE_POSITIVE = False


def wilson_lb(successes: int, n: int, z: float = _Z90) -> float:
    """Wilson score lower bound for a binomial proportion (one-sided)."""
    if n <= 0:
        return 0.0
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def r_lb(stats: dict) -> float:
    """Wilson lower bound B(D, S) of the loss rate among samples the reference skill solves."""
    return wilson_lb(len(stats.get("severe_ids", [])),
                     max(1, stats.get("n_parent_success", 0)))


def flip_balance_lb(stats: dict) -> float:
    """Wilson lower bound B(D, D+U) of the share of losses among losses and
    improvements. A value above 0.5 indicates that losses form a majority;
    0.0 when there are neither."""
    down = len(stats.get("severe_ids", []))
    up = len(stats.get("severe_up_ids", []))
    if down + up == 0:
        return 0.0
    return wilson_lb(down, down + up)


def combine_rounds(s1: dict, s2: dict) -> dict:
    """Combine the paired statistics of two disjoint screening sets."""
    n = s1["n_pairs"] + s2["n_pairs"]
    if n == 0:
        return dict(s1)
    out = dict(s2)
    for k in ("G", "G_plus", "G_minus", "aux_mean_delta"):
        out[k] = (s1[k] * s1["n_pairs"] + s2[k] * s2["n_pairs"]) / n
    for k in ("W", "L"):
        out[k] = s1[k] + s2[k]
    out["n_pairs"] = n
    out["n_neq"] = out["W"] + out["L"]
    ns = s1["n_parent_success"] + s2["n_parent_success"]
    sev = len(set(s1["severe_ids"]) | set(s2["severe_ids"]))
    out["R"] = sev / max(1, ns)
    out["n_parent_success"] = ns
    out["severe_ids"] = sorted(set(s1["severe_ids"]) | set(s2["severe_ids"]))
    out["severe_up_ids"] = sorted(set(s1.get("severe_up_ids", [])) | set(s2.get("severe_up_ids", [])))
    out["guard_severe_ids"] = sorted(set(s1["guard_severe_ids"]) | set(s2["guard_severe_ids"]))
    out["eps"] = max(s1["eps"], s2["eps"])
    out["eps_guard"] = min(0.01, out["eps"])
    out["aux_guardrail_ok"] = out["aux_mean_delta"] >= -out["eps_guard"]
    out["aux_guardrail_catastrophic"] = out["aux_mean_delta"] < -2 * out["eps_guard"]
    out["coverage"] = min(s1["coverage"], s2["coverage"])
    return out


def decide(stats: dict, *, win_categories: int, loss_categories: int,
           patch_applied_ok: bool, final_round: bool) -> dict:
    """win/loss_categories: number of distinct strata among the samples the
    candidate wins / loses. final_round: True when this is the second screening
    set for the candidate, so that no further screening is allowed."""
    G, W, L, R = stats["G"], stats["W"], stats["L"], stats["R"]
    eps, n_neq = stats["eps"], stats["n_neq"]
    guards_severe = len(stats["guard_severe_ids"])
    RLB = r_lb(stats)             # B(D, S): losses among samples the current skill solves
    FLB = flip_balance_lb(stats)  # B(D, D+U): losses relative to improvements
    reasons = []

    # ---- immediate reject ----
    if not patch_applied_ok:
        return {"decision": "reject", "reasons": ["safety veto: patch not mechanically applicable"]}
    if stats["coverage"] < 0.90:
        return {"decision": "invalid", "reasons": [f"paired coverage {stats['coverage']} < 0.90"]}
    if guards_severe >= 2:
        return {"decision": "reject", "reasons": [f"safety veto: {guards_severe} severe regressions in random guards"]}
    if RLB > 0.50:
        return {"decision": "reject",
                "reasons": [f"safety veto: catastrophic regression LB {RLB:.3f} > 0.50 (R={R:.3f})"]}
    if FLB > 0.65:
        return {"decision": "reject",
                "reasons": [f"safety veto: down-flips dominate (flip-balance LB {FLB:.3f} > 0.65)"]}

    evidence_ready = n_neq >= 6

    # ---- accept ----
    commit_ok = (evidence_ready and G >= eps and W >= L and FLB <= 0.50
                 and RLB <= 0.50 and guards_severe <= 1
                 and not stats["aux_guardrail_catastrophic"])
    soft_guard_violation = commit_ok and not stats["aux_guardrail_ok"]
    if commit_ok and not soft_guard_violation:
        return {"decision": "commit",
                "reasons": [f"G={G:.4f}>=eps={eps:.4f}, W={W}>=L={L}, flipLB={FLB:.3f}<=0.50 (R={R:.3f})"]}
    if soft_guard_violation and not final_round:
        return {"decision": "probation",
                "reasons": [f"primary commit gate passed but guardrail mean delta "
                            f"{stats['aux_mean_delta']:.4f} < -{stats['eps_guard']:.4f}; one more round"]}

    # ---- save without acceptance ----
    if evidence_ready:
        w_min = max(2, math.ceil(0.15 * n_neq))
        branch_ok = (W >= w_min and L >= w_min
                     and win_categories >= 2 and loss_categories >= 2
                     and stats["G_plus"] >= eps / 2 and stats["G_minus"] >= eps / 2
                     and FLB <= 0.60 and RLB <= 0.50 and guards_severe <= 1)
        if branch_ok:
            return {"decision": "branch",
                    "reasons": [f"heterogeneous: W={W},L={L}>=w_min={w_min}, "
                                f"G+={stats['G_plus']:.4f}, G-={stats['G_minus']:.4f}"]}

    # ---- reject ----
    if evidence_ready:
        if G <= -eps and L > W:
            return {"decision": "reject", "reasons": [f"G={G:.4f}<=-eps, L={L}>W={W}"]}
        if L - W >= 3:
            return {"decision": "reject", "reasons": [f"L-W={L-W}>=3 with n_neq={n_neq}"]}
    if W == 0 and L >= 3:
        return {"decision": "reject", "reasons": [f"no candidate win and parent has {L} wins"]}

    # ---- inconclusive ----
    if final_round:
        if (ARCHIVE_POSITIVE and G > 0 and W >= L
                and FLB <= 0.50 and RLB <= 0.50 and guards_severe <= 1):
            return {"decision": "branch",
                    "reasons": [f"positive-in-band archived for tournament "
                                f"(G={G:.4f}, n_neq={n_neq})"]}
        return {"decision": "insufficient_evidence",
                "reasons": [f"after 2 rounds: n_neq={n_neq}, G={G:.4f} in (-eps,eps)={eps:.4f} band"]}
    return {"decision": "probation",
            "reasons": [f"n_neq={n_neq}<6 or |G|<eps: need a second disjoint impact set"]}
