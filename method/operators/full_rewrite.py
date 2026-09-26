"""Full-rewrite revision form (F4).

One step turns the execution feedback on a few samples into critiques of the
current skill and asks the optimizer LLM for a rewritten skill that keeps the
output-format specification. Used by I2 (as one of its forms) and by I3.
"""
from method.common import util
from method.config import OPTIMIZER_MODEL
from method.config import _truncate
from method.common.execution import client_for


# ---------------------------------------------------------------------------
# Prompt templates. The critique and rewrite templates below follow TextGrad
# (Yuksekgonul et al., 2024; autograd/llm_backward_prompts.py and
# optimizer/optimizer_prompts.py); the {}-fields are filled here.
BATCH_SIZE = 3

GLOSSARY_TEXT_BACKWARD = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <OBJECTIVE_FUNCTION>: The objective of the optimization task.
# - <VARIABLE>: Specifies the span of the variable.
# - <ROLE>: The role description of the variable."""

BACKWARD_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves a given text (i.e. the variable). You are the gradient (feedback) engine. "
    "Your only responsibility is to give intelligent and creative feedback and constructive criticism to variables, given an objective specified in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags. "
    "The variables may be solutions to problems, prompts to language models, code, or any other text-based variable. "
    "Pay attention to the role description of the variable, and the context in which it is used. You should assume that the variable will be used in a similar context in the future. "
    "Only provide strategies, explanations, and methods to change in the variable. DO NOT propose a new version of the variable, that will be the job of the optimizer. Your only job is to send feedback and criticism (compute 'gradients'). "
    "For instance, feedback can be in the form of 'Since language models have the X failure mode...', 'Adding X can fix this error because...', 'Removing X can improve the objective function because...', 'Changing X to Y would fix the mistake ...', that gets at the downstream objective.\n"
    "If a variable is already working well (e.g. the objective function is perfect, an evaluation shows the response is accurate), you should not give feedback.\n"
    f"{GLOSSARY_TEXT_BACKWARD}")

CONVERSATION_TEMPLATE = (
    "<LM_SYSTEM_PROMPT> {system_prompt} </LM_SYSTEM_PROMPT>\n\n"
    "<LM_INPUT> {prompt} </LM_INPUT>\n\n"
    "<LM_OUTPUT> {response_value} </LM_OUTPUT>\n\n"
)

CONVERSATION_START_INSTRUCTION_CHAIN = (
    "You will give feedback to a variable with the following role: <ROLE> {variable_desc} </ROLE>. "
    "Here is a conversation with a language model (LM):\n\n"
    "{conversation}"
)
OBJECTIVE_INSTRUCTION_CHAIN = (
    "This conversation is part of a larger system. The <LM_OUTPUT> was later used as {response_desc}.\n\n"
    "<OBJECTIVE_FUNCTION>Your goal is to give feedback to the variable to address the following feedback on the LM_OUTPUT: {response_gradient} </OBJECTIVE_FUNCTION>\n\n"
)
EVALUATE_VARIABLE_INSTRUCTION = (
    "We are interested in giving feedback to the {variable_desc} "
    "for this conversation. Specifically, give feedback to the following span "
    "of text:\n\n<VARIABLE> "
    "{variable_short} </VARIABLE>\n\n"
    "Given the above history, describe how the {variable_desc} "
    "could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent.\n\n"
)

GLOSSARY_TEXT = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <FEEDBACK>: The feedback to the variable.
# - <CONVERSATION>: The conversation history.
# - <FOCUS>: The focus of the optimization.
# - <ROLE>: The role description of the variable."""

OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "You will be asked to creatively and critically improve prompts, solutions to problems, code, or any other text-based variable. "
    "You will receive some feedback, and use the feedback to improve the variable. "
    "The feedback may be noisy, identify what is important and what is correct. "
    "Pay attention to the role description of the variable, and the context in which it is used. "
    "This is very important: You MUST give your response by sending the improved variable between <IMPROVED_VARIABLE> {improved variable} </IMPROVED_VARIABLE> tags. "
    "The text you send between the tags will directly replace the variable.\n\n"
    f"{GLOSSARY_TEXT}"
)

TGD_PROMPT_PREFIX = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>.\n\n"
    "The variable is the text within the following span: <VARIABLE> {variable_short} </VARIABLE>\n\n"
    "Here is the context and feedback we got for the variable:\n\n"
    "<CONTEXT>{variable_grad}</CONTEXT>\n\n"
    "Improve the variable ({variable_desc}) using the feedback provided in <FEEDBACK> tags.\n"
)
TGD_PROMPT_SUFFIX = (
    "Send the improved variable "
    "in the following format:\n\n<IMPROVED_VARIABLE>{{the improved variable}}</IMPROVED_VARIABLE>\n\n"
    "Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else."
)
GRADIENT_TEMPLATE = (
    "Here is a conversation:\n\n<CONVERSATION>{context}</CONVERSATION>\n\n"
    "This conversation is potentially part of a larger system. The output is used as {response_desc}\n\n"
    "Here is the feedback we got for {variable_desc} in the conversation:\n\n<FEEDBACK>{feedback}</FEEDBACK>\n\n"
)
# --- end verbatim templates -------------------------------------------------

VARIABLE_DESC = ("structured system prompt to a somewhat capable language model "
                 "that specifies the behavior and strategies for the task")
RESPONSE_DESC = "the answer to be evaluated by the task's official metric"


def task_input_summary(dataset, task):
    if dataset == "docvqa":
        return f"[Document page image] Question: {task['question']}"
    if dataset == "mind2web":
        r = task["record"]
        return (f"Website task on {r.get('website')}: {r.get('task')} "
                f"({len(r.get('steps', []))} teacher-forced steps, 20 candidate elements each)")
    if dataset == "livemath":
        ch = " | ".join(f"{c['label']}. {_truncate(c['text'], 120)}"
                        for c in task.get("choices", []))
        return (f"Research-math MCQ: {_truncate(task['question'], 900)}\n"
                f"Choices: {ch}")
    t = task["task_obj"]
    return f"Spreadsheet instruction ({t.instruction_type}): {_truncate(t.instruction, 1200)}"


def result_summary(dataset, rec):
    if "error" in rec and "primary" not in rec:
        return f"EXECUTION ERROR: {rec['error']}", 0.0
    score = float(rec.get("primary", 0.0))
    if dataset == "docvqa":
        out = _truncate(rec.get("response_text", ""), 1500)
        fb = (f"ANLS={rec.get('anls', score):.3f}; extracted answer {rec.get('predicted_answer')!r}; "
              f"gold answers {rec.get('gold_answers')}")
        return out + "\n\n[Evaluation] " + fb, score
    if dataset == "mind2web":
        lines = []
        for i, s in enumerate(rec.get("per_step", [])):
            ok = "correct-element" if s.get("element_correct") else "WRONG-element"
            lines.append(f"step{i}: ({s.get('operation')},{s.get('element_id')},"
                         f"{_truncate(s.get('value'), 30)}) -> {ok}")
        fb = f"task element-accuracy {rec.get('ea', 0):.3f}, step-success {rec.get('ssr', 0):.3f}"
        return "\n".join(lines) + "\n\n[Evaluation] " + fb, score
    if dataset == "livemath":
        out = _truncate(rec.get("response_tail", ""), 800)
        fb = (f"predicted label {rec.get('predicted_label')!r}; "
              f"{'correct' if score >= 1 else 'INCORRECT'}; parse-ok={rec.get('parsed')}")
        return out + "\n\n[Evaluation] " + fb, score
    errs = []
    for cr in rec.get("case_results", []):
        for e in (cr.get("errors") or [])[:3]:
            errs.append(_truncate(e, 150))
    out = _truncate(rec.get("code", rec.get("response_text", "")), 2000)
    fb = (f"hard_pass={rec.get('hard_pass')} cell_accuracy={rec.get('cell_accuracy')} "
          + (f"errors: {' | '.join(errs)}" if errs else "all cases passed"))
    return out + "\n\n[Evaluation] " + fb, score


def optimizer_call(system, prompt, tag):
    client = client_for("optimizer/textgrad")
    resp = client.converse(
        OPTIMIZER_MODEL, [{"role": "user", "content": [{"text": prompt}]}],
        system=system, max_tokens=16384, temperature=0.0, seed=42,
        tags={"stage": tag})
    return resp["text"]


def extract_improved(text):
    lo, hi = text.rfind("<IMPROVED_VARIABLE>"), text.rfind("</IMPROVED_VARIABLE>")
    if lo == -1 or hi == -1 or hi <= lo:
        raise ValueError("optimizer response missing <IMPROVED_VARIABLE> tags")
    return text[lo + len("<IMPROVED_VARIABLE>"):hi].strip()


# ---------------------------------------------------------------------------
# One F4 step: critiques of a few failed samples, then a full rewrite.
REFINE_VAL_N = 128
GATE_EVERY = 4

# The rewrite must keep the skill's output-format specification unchanged;
# the parser downstream depends on it.
FORMAT_GUARD = (
    "\n\nHARD CONSTRAINT: the improved variable MUST reproduce the original "
    "variable's output-format specification EXACTLY, character-for-character "
    "(the action-space list and the final answer-line format, including every "
    "bracket and backtick). A downstream parser depends on it. Only the "
    "reasoning/guidance sections may be changed.")


def refine_step(ds, artifact: str, batch_tasks, batch_res, variant: int = 0, run_salt: str = "",
                *, form="F4", ev=None, history="", generation=None) -> str | None:
    """One textual-gradient step: critiques on a few samples, then a full
    rewrite. Returns the rewritten skill or None if no <IMPROVED_VARIABLE>
    block is found. `variant` rotates the critique count (3/4/5) so that
    repeating the same (skill, batch) pair yields a new rewrite rather than a
    cached one."""
    n_crit = 3 + (variant % 3)
    fails = [t for t in batch_tasks
             if (batch_res.get(t["task_id"], {}).get("primary") is not None)][:n_crit]
    grads = []
    for t in fails:
        rec = batch_res.get(t["task_id"]) or {}
        from method.state import safe_summary
        out, score = safe_summary(ds, rec)
        conversation = CONVERSATION_TEMPLATE.format(
            system_prompt=artifact, prompt=task_input_summary(ds, t), response_value=out)
        bp = (CONVERSATION_START_INSTRUCTION_CHAIN.format(
                variable_desc=VARIABLE_DESC, conversation=conversation)
              + OBJECTIVE_INSTRUCTION_CHAIN.format(
                response_desc=RESPONSE_DESC,
                response_gradient=f"official task score {score:.3f} (1.0 is perfect); see the [Evaluation] block inside LM_OUTPUT")
              + EVALUATE_VARIABLE_INSTRUCTION.format(
                variable_desc=VARIABLE_DESC, variable_short=artifact[:12000]))
        grads.append((conversation, optimizer_call(BACKWARD_SYSTEM_PROMPT, bp, f"v14_backward{run_salt}")))
    variable_grad = "".join(
        GRADIENT_TEMPLATE.format(context=c, response_desc=RESPONSE_DESC,
                                 variable_desc=VARIABLE_DESC, feedback=g)
        for c, g in grads)
    # Content specification shared with Direct Revision (I1).
    from method.operators import optimizer_llm
    spec = optimizer_llm.V25_CONTENT_SPEC if optimizer_llm.CONTENT_V25 else ""
    if form != "F4":
        from method.operators.iterative_refinement import local_update
        return local_update(artifact, variable_grad, form, ev or {}, history,
                            spec, run_salt, generation)
    tgd = (TGD_PROMPT_PREFIX.format(variable_desc=VARIABLE_DESC, variable_short=artifact,
                                    variable_grad=variable_grad) + TGD_PROMPT_SUFFIX
           + FORMAT_GUARD + spec)
    if history:
        tgd += "\n\n" + history
    try:
        text = extract_improved(optimizer_call(OPTIMIZER_SYSTEM_PROMPT, tgd, f"v14_update{run_salt}"))
        if generation is not None:
            generation.update(mechanically_valid=True, source_valid=True,
                              source_task_ids=[], generator="original_tgd_full_rewrite")
        return text
    except ValueError:
        if generation is not None:
            generation.update(mechanically_valid=False, source_valid=True,
                              failure_stage="rewrite_extraction")
        return None


def refine_val_ids(train_ids, used_ids: set, salt: str):
    pool = [t for t in train_ids if t not in used_ids]
    pool.sort(key=lambda t: util.sha256_str(f"v14val|{salt}|{t}"))
    return pool[:REFINE_VAL_N]
