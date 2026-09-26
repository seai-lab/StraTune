"""Offline behavioral checks of candidate generation, evaluation history, and refinement logic (no network)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import socket


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT
sys.path[:0] = [str(CODE)]


def no_network(*args, **kwargs):
    raise AssertionError("Smoke tests must not access the network")


socket.socket.connect = no_network
socket.create_connection = no_network
from method.operators import full_rewrite
from method.state import RunState
from method.state import BudgetMeter
from method import state
from method.operators import iterative_refinement
from method.operators import selection


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.st = RunState(self.temp.name, "Seed skill. Final answer: <label>")
        self.st.run_id = "smoke_run"
        self.st.sampling_id = "paired_smoke"
        self.st.d["_ds"] = "livemath"
        self.tasks = [{"task_id": str(i), "question": f"Question {i}", "choices": []} for i in range(20)]
        self.by_id = {t["task_id"]: t for t in self.tasks}
        self.budget = BudgetMeter("livemath", 10000, self.st.d)
        self.ev = {"n_fail": 2, "n_batch": 4, "fail_lines": ["wrong label"],
                   "gold_lines": [], "offered_gold_ids": [], "lesson_ids": [],
                   "contrast_lines": [], "traj_lines": [], "longitudinal": ""}
        self.executor = SimpleNamespace(run=lambda t, s: {
            "primary": 0.0 if s.startswith("Private") else 1.0,
            "response_tail": f"output under {s}", "predicted_label": "A", "parsed": True})
        self.ctx = {"batch_ids": [str(i) for i in range(4)],
                    "train_ids": list(self.by_id), "tasks_by_id": self.by_id,
                    "executor": self.executor, "workers": 2, "budget": self.budget,
                    "n_slice": 8, "rnd": 3}

    def test_local_long_edits_are_not_trimmed(self):
        text = "\n".join(f"Step {i} retain this full instruction" for i in range(1500))
        raw = {"source_task_ids": [], "patch": {"edits": [
            {"operation": "append", "old_content": "", "new_content": text}]}}
        info = {}
        with patch.object(full_rewrite, "optimizer_call", return_value=json.dumps(raw)):
            result = iterative_refinement.local_update("seed", "critique", "F3", self.ev, "", "", "s", info)
        self.assertIn(text, result)
        self.assertTrue(info["mechanically_valid"])
        self.assertTrue(info["source_valid"])

    def test_f2_availability_and_exact_grounding(self):
        self.assertNotIn("F2", iterative_refinement.available_forms(self.ev))
        ev = dict(self.ev, gold_lines=["[1] Input -> GOLD"], offered_gold_ids=["1"])
        self.assertIn("F2", iterative_refinement.available_forms(ev))
        def generate(line, ids):
            raw = {"source_task_ids": ids, "patch": {"edits": [
                {"operation": "append", "old_content": "", "new_content": "### Worked examples\n" + line}]}}
            info = {}
            with patch.object(full_rewrite, "optimizer_call", return_value=json.dumps(raw)):
                result = iterative_refinement.local_update("seed", "", "F2", ev, "", "", "", info)
            return result, info
        self.assertIsNotNone(generate("[1] Input -> GOLD", ["1"])[0])
        for line, ids in [("[1] Input -> invented", ["1"]), ("[2] Input -> GOLD", ["2"]),
                          ("[1] Input -> GOLD", []), ("[1] Input -> GOLD\nExtra fabricated case", ["1"])]:
            result, info = generate(line, ids)
            self.assertIsNone(result)
            self.assertEqual(info["failure_stage"], "example_grounding")

    def test_bad_trailing_edit_rejects_whole_candidate(self):
        raw = {"source_task_ids": [], "patch": {"edits": [
            {"operation": "append", "old_content": "", "new_content": "valid"},
            {"operation": "replace", "old_content": "absent", "new_content": "bad"}]}}
        info = {}
        with patch.object(full_rewrite, "optimizer_call", return_value=json.dumps(raw)):
            result = iterative_refinement.local_update("seed", "", "F1", self.ev, "", "", "", info)
        self.assertIsNone(result)
        self.assertEqual(info["failure_stage"], "patch_validation")

    def test_existing_strategy_call_selects_i2_form(self):
        calls = []
        def choose(prompt, **kw):
            calls.append(prompt)
            return {"iteration": "I2", "form": "F3", "why": "procedure needed"}
        with patch.object(selection, "I2_FORM_MODE", "free"), patch.object(selection.optimizer_llm, "_call_json", choose):
            choice = selection.declare("livemath", self.st, self.ev, self.ctx, ["I2"], lambda _: None)
        self.assertEqual(choice["form"], "F3")
        self.assertEqual(len(calls), 1)
        self.assertIn(self.st.skill_text(self.st.d["champion"]), calls[0])
        self.assertNotIn("F2: Worked", calls[0])

    def test_f4_uses_native_template_and_output_parser(self):
        prompts = []
        def respond(system, prompt, tag):
            prompts.append((system, prompt, tag))
            return "<IMPROVED_VARIABLE>complete rewrite</IMPROVED_VARIABLE>"
        with patch.object(full_rewrite, "optimizer_call", respond):
            value = full_rewrite.refine_step("livemath", "old skill", [], {}, form="F4",
                                   history="SHARED HISTORY", generation={})
        self.assertEqual(value, "complete rewrite")
        self.assertEqual(prompts[0][0], full_rewrite.OPTIMIZER_SYSTEM_PROMPT)
        self.assertIn(full_rewrite.TGD_PROMPT_SUFFIX, prompts[0][1])
        self.assertIn(full_rewrite.FORMAT_GUARD, prompts[0][1])
        self.assertNotIn("Return ONLY JSON", prompts[0][1])
        self.assertTrue(prompts[0][2].startswith("v14_update"))

    def test_cache_missing_output_is_unknown_not_empty(self):
        value, score = state.safe_summary("spreadsheetbench", {"primary": 0.0, "score_only": True})
        self.assertEqual(score, 0.0)
        self.assertIn("not available", value)
        self.assertNotIn("all cases passed", value)
        self.assertTrue(state.has_output("livemath", {"primary": 0, "response_tail": ""}))
        self.assertFalse(state.has_output("livemath", {"primary": 0}))

    def test_real_records_survive_state_reload(self):
        vid = self.st.d["champion"]
        rec = {"primary": 0.0, "response_tail": "", "predicted_label": None, "parsed": False}
        self.st.record_scores("livemath", vid, {"0": rec})
        self.st.save()
        reloaded = RunState(self.temp.name, "ignored")
        record = reloaded.known_records(vid, "livemath")["0"]
        self.assertIn("response_tail", record)
        self.assertEqual(record["response_tail"], "")
        self.assertEqual(record["primary"], 0)
        self.assertTrue(record["cached"])

    def test_intermediate_feedback_reexecutes_exact_base_and_bills(self):
        parent = self.st.d["champion"]
        private = self.st.add_version("Private skill", parent, {}, 1)
        self.st.record_scores("livemath", parent, {"0": {"primary": 1, "response_tail": "champion"}})
        results = state.records_for_feedback("livemath", self.st, private, self.tasks,
                                                self.executor, 2, self.budget, 3)
        self.assertEqual(self.budget.billed, 3)
        self.assertTrue(all(r["primary"] == 0 for r in results.values()))
        state.records_for_feedback("livemath", self.st, private, self.tasks,
                                      self.executor, 2, self.budget, 3)
        self.assertEqual(self.budget.billed, 3)

    def test_reference_skill_and_samples_are_keyed(self):
        parent = self.st.d["champion"]
        private = self.st.add_version("Private skill", parent, {}, 1)
        iterative_refinement._gen(self.st)
        args = (self.by_id, self.executor, 2, self.budget)
        a = state.reference("livemath", self.st, parent, ["0", "1"], *args)
        b = state.reference("livemath", self.st, private, ["0", "1"], *args)
        c = state.reference("livemath", self.st, parent, ["0"], *args)
        self.assertEqual((a[0], b[0], c[0]), (1, 0, 1))
        self.assertEqual(len({a[2], b[2], c[2]}), 3)
        self.assertEqual(len(c[1]), 1)

    def test_zero_gain_can_advance_without_relaxed_rule(self):
        gen = iterative_refinement._gen(self.st)
        vid = self.st.add_version("new", self.st.d["champion"], {}, 1)
        gen.update(val_score=1.0, window=[{"score": 1.0, "base_score": 1.0, "vtext": "new"}])
        iterative_refinement._submit_window(gen, self.st, self.budget, 1,
                                lambda *a: ("branch", {"G": 0.0, "eps": 0.01}, False, vid),
                                lambda _: None, seats=1)
        self.assertEqual(gen["base_vid"], vid)
        self.assertEqual(gen["base_G"], 0.0)
        self.assertIsNone(gen["val_score"])

    def test_i2_attempts_and_submission_counts_are_separate(self):
        e = state.begin(self.st, "F3", 2, "base", "sha", ["1"])
        state.update(self.st, e, mechanically_valid=False, failure_stage="patch_validation")
        rendered = state.render_form_history(self.st)
        self.assertIn("F3: attempts=1, valid_generation=0", rendered)
        self.assertNotIn("not tried yet this run", rendered)
        state.submitted(self.st, "uid", {"attempt_path": [e["attempt_id"]]})
        for _ in range(2):
            state.outcome(self.st, "uid", "screening", {"decision": "branch", "mean_gain": 0.1})
        self.assertEqual(len(e["submissions"]), 1)
        self.assertIsNone(e["internal"])

    def test_saved_candidates_are_visible_and_case_limit_is_shared(self):
        book = self.st.d.setdefault("adaptive_case_history", {})
        for i in ("I1", "I2", "I3"):
            for j in range(5):
                uid = f"{i}_{j}"
                book[uid] = {
                    "uid": uid, "strategy": i, "source_round": j, "problem": "p", "change": "c",
                    "form": "F3", "form_path": ["F3"], "form_explicitly_selected": True,
                    "result": "saved_without_acceptance" if j % 2 else "rejected",
                    "screening": {"decision": "branch", "n_pairs": 8, "mean_gain": 0.1,
                                  "wins": 4, "losses": 2, "improved": [{"input": "good"}],
                                  "regressed": [{"input": "bad"}]},
                }
        selected = state.select_cases(self.st)
        self.assertEqual(len(selected), 6)
        rendered = state.render_cases(self.st)
        self.assertIn("saved_without_acceptance", rendered)
        self.assertIn("good", rendered)
        self.assertIn("bad", rendered)

    def test_i2_end_to_end_step_has_aligned_internal_evidence(self):
        self.st._round_evidence = self.ev
        self.st._i2_form = "F3"
        private = self.st.add_version("Private skill", self.st.d["champion"], {}, 1)
        iterative_refinement._gen(self.st)["base_vid"] = private
        raw = {"patch": {"edits": [{"operation": "append", "old_content": "", "new_content": "procedure"}]},
               "source_task_ids": []}
        calls = []
        def respond(system, prompt, tag):
            calls.append((prompt, tag))
            return json.dumps(raw) if tag.startswith("i2_local_update") else "critique"
        with patch.object(full_rewrite, "optimizer_call", respond):
            iterative_refinement.rewriter_round("livemath", self.st, self.executor, self.ctx["batch_ids"],
                                    {"0": {"primary": 1, "response_tail": "WRONG CHAMPION RECORD"}},
                                    self.by_id, list(self.by_id), self.budget, 2, 3, 8,
                                    lambda *a: None, lambda _: None)
        entry = next(iter(self.st.d["i2_attempts"].values()))
        self.assertTrue(entry["valid_generation"])
        self.assertEqual(entry["generation_base"], private)
        self.assertEqual(entry["internal"]["base_score"], 0)
        self.assertEqual(entry["internal"]["mean_gain"], 0)
        self.assertEqual(sum(t.startswith("i2_local_update") for _, t in calls), 1)
        self.assertTrue(all("WRONG CHAMPION RECORD" not in p for p, _ in calls))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "network_calls": 0, "model_responses": "mocked"}
    (Path(__file__).parent / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(not result.wasSuccessful())
