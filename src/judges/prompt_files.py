"""Loader for the frozen prompt files in `prompts/`.

The prompt text lives in files, not in Python, for one reason: ground rule 1
says a stimulus must render byte-identically across runs and across models, and
a file can be hashed and diffed where an f-string buried in a module cannot.
Each file is:

    # comment lines, dropped
    <<<SYSTEM>>>
    ...system half...
    <<<USER>>>
    ...user half, with {question} {evidence} {answer} placeholders...
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

SYSTEM_MARK = "<<<SYSTEM>>>"
USER_MARK = "<<<USER>>>"

#: Which pair of words the judge is told to emit. The verdict contrast is read
#: off this, so it is part of the prompt's identity, not a runner option.
CONVENTIONS = {
    "J1": ("CORRECT", "INCORRECT"),
    "J2a": ("CORRECT", "WRONG"),
    "J2b": ("CORRECT", "WRONG"),
    "J2c": ("CORRECT", "WRONG"),
}


@dataclass(frozen=True)
class JudgePrompt:
    name: str
    system: str
    user_template: str
    positive: str
    negative: str
    sha256: str

    def render(self, *, question: str, evidence: str, answer: str) -> str:
        return self.user_template.format(
            question=question, evidence=evidence, answer=answer
        )


def load(name: str) -> JudgePrompt:
    path = PROMPTS_DIR / f"{name}.txt"
    raw = path.read_bytes()
    text = raw.decode()
    body = "\n".join(
        line for line in text.splitlines() if not line.startswith("#")
    )
    if SYSTEM_MARK not in body or USER_MARK not in body:
        raise ValueError(f"{path} is missing {SYSTEM_MARK} or {USER_MARK}")
    _, rest = body.split(SYSTEM_MARK, 1)
    system, user = rest.split(USER_MARK, 1)
    positive, negative = CONVENTIONS[name]
    return JudgePrompt(
        name=name,
        system=system.strip(),
        user_template=user.strip() + "\n",
        positive=positive,
        negative=negative,
        # Hash the file as it sits on disk, comments included: the comments say
        # where the wording came from, so a change to them is a provenance
        # change worth noticing.
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def placeholders(template: str) -> set[str]:
    return set(re.findall(r"\{(\w+)\}", template))
