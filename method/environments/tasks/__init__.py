"""Shared benchmark environments (a single evaluator per benchmark, used for
both training and test).

The scorers are fixed; each environment records its version string
(``VERSION``) in every result. Modules:

* ``method.environments.tasks.docvqa``              -- DocVQAEnv, official ANLS
* ``method.environments.tasks.spreadsheet_codegen`` -- SpreadsheetCodegenEnv,
  SkillOpt one-code/all-cases protocol +
  ``method.environments.metrics.spreadsheet_checks``
* ``method.environments.tasks.mind2web``            -- M2WEnv, thin glue over the
  teacher-forced evaluator in ``method.environments.mind2web_prompting``
* ``method.environments.tasks.livemath``            -- LiveMathEnv, MCQ label
  accuracy

Import the submodules directly; heavy deps stay lazy per module.
"""

from .docvqa import DocVQAEnv
from .mind2web import M2WEnv
from .spreadsheet_codegen import SpreadsheetCodegenEnv

__all__ = ["DocVQAEnv", "M2WEnv", "SpreadsheetCodegenEnv"]
