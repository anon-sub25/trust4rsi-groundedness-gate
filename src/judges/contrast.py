"""Resolving the verdict contrast when the two words are not single tokens.

`judgegeom.local_judge.resolve_contrast_pair` demands that both verdict words
encode to exactly one token, and raises otherwise. That is right for its own
CORRECT/INCORRECT prompts, where a leading-space variant always rescues the
pair. It does not carry over to J2a: Qwen2.5 encodes "CORRECT" as ("COR",
"RECT") and "WRONG" as ("W", "RONG"), so neither side is single-token and no
capitalisation variant matches the prompt's own uppercase wording.

The fix is to score the *first* token of each verdict rather than the whole
word. That is sound exactly when the two first tokens differ, because the only
two strings the prompt permits then diverge at the very position being scored:
p(COR) vs p(W) at position 0 is p(CORRECT) vs p(WRONG). It would be unsound if
the two words shared a first token (the scored position would carry no verdict
information), so that case raises.
"""

from __future__ import annotations

from typing import Callable, Sequence


def resolve_first_token_contrast(
    encode: Callable[[str], Sequence[int]], positive: str, negative: str
) -> tuple[int, int]:
    ids_positive = list(encode(positive))
    ids_negative = list(encode(negative))
    if not ids_positive or not ids_negative:
        raise ValueError(f"empty encoding for {positive!r}/{negative!r}")
    first_positive, first_negative = ids_positive[0], ids_negative[0]
    if first_positive == first_negative:
        raise ValueError(
            f"{positive!r} and {negative!r} share a first token "
            f"({first_positive}); the scored position carries no verdict "
            "information. Pick verdict words that diverge immediately."
        )
    return first_positive, first_negative
