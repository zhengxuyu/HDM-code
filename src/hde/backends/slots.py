"""Compile independent question branches sharing only the state prefix."""

from dataclasses import dataclass
from itertools import product
from string import ascii_uppercase

from hde.core.prompts import build_decision_messages
from hde.core.prompts_utils import question_suffix
from hde.formats import SLOT_FORMAT  # noqa: F401


@dataclass(frozen=True)
class CompiledSlots:
    input_ids: list[list[int]]
    positions: list[int]
    candidates: list[list[int]]
    slot_texts: list[str]
    slot_ids: list[list[int]]
    prefix_text: str
    prefix_length: int
    candidate_codes: list[list[str]]


def candidate_codes(tokenizer, count):
    """Deterministic letter codes, accepting only exact one-token continuations."""
    boundary = "Answer:"
    prefix = tokenizer.encode(boundary, add_special_tokens=False)
    codes, seen = [], set()
    for width in range(1, 4):
        for letters in product(ascii_uppercase, repeat=width):
            code = "".join(letters)
            ids = tokenizer.encode(boundary + " " + code, add_special_tokens=False)
            if len(ids) == len(prefix) + 1 and ids[:-1] == prefix and ids[-1] not in seen:
                codes.append(code)
                seen.add(ids[-1])
                if len(codes) == count:
                    return codes
    raise ValueError(f"Tokenizer cannot supply {count} distinct single-token answer codes")


def boundary_candidate_ids(tokenizer, text, codes):
    """Each code's vocabulary id as the model would produce it after `text`.

    One function, because there were two and they disagreed. This one is what
    `compile_prefix_slots` has always done -- encode `text + " " + code` and take
    the token the space binds to -- while `SGLangScorer._candidate_ids` encoded the
    bare code. On a BPE tokenizer `" A"` and `"A"` are different entries, so the two
    backends read different rows of the same logit vector and neither failed: both
    ids exist, both are single tokens, and nothing tied them to each other.

    The space-bound one is correct, and `candidate_codes` is the proof: it chooses
    the code set by requiring `"Answer:" + " " + code` to add exactly one token, so
    the codes are single tokens *in that position*. After a colon the model puts its
    mass on `" A"`; the bare `"A"` is a different token it has little reason to emit
    there. Measured consequence: over MMLU-Pro's stride-8 1500 on the 27B the two
    backends' coded readouts agreed on only 78.1% of questions, with systematic
    shifts rather than near-ties (code-reviewer, from 28037's and 28284's rows).
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    ids = []
    for code in codes:
        appended = tokenizer.encode(text + " " + code, add_special_tokens=False)
        if len(appended) != len(tokens) + 1 or appended[:-1] != tokens:
            raise ValueError(
                f"Tokenizer must encode internal code {code!r} as one token at the answer boundary"
            )
        ids.append(appended[-1])
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate token collision")
    return ids


def compile_slots(tokenizer, state, schema, max_input_tokens=16384, code_offset=0):
    prefix = tokenizer.apply_chat_template(
        build_decision_messages(state, schema),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    return compile_prefix_slots(tokenizer, schema, prefix_ids, prefix, max_input_tokens,
                                code_offset=code_offset)


def rotated(codes, offset):
    """The same codes, moved along by `offset`, so slot i is labelled code i+offset.

    Nothing about a question changes: the options keep their order, the readout
    keeps reading logits in slot order, and only the letter printed beside each
    option moves. That is what separates the two readings of the readout's
    preference, which every run so far has confounded. `candidate_codes` hands
    out A, B, C in slot order, so "prefers the second slot" and "prefers the
    letter B" name the same answer at every width and over any number of
    questions -- the distribution over letters is the distribution over slots.
    With the letters moved, a preference that follows the letter moves with it
    and one that follows the slot stays where it was.
    """
    if not offset:
        return list(codes)
    turn = offset % len(codes)
    return list(codes[turn:]) + list(codes[:turn])


def compile_prefix_slots(tokenizer, schema, prefix_ids, prefix="", max_input_tokens=16384,
                         code_offset=0):
    """Share label-boundary checks between text and processor-expanded image prefixes.

    `code_offset` rotates the letters against the slots; see `rotated`. It is a
    diagnostic, not a fix: a run that sets it is asking what the readout is
    reading, and the answer belongs beside the number it produces.
    """
    prefix_length = len(prefix_ids)
    rows = []
    positions, candidates, texts, slot_ids = [], [], [], []
    codes = candidate_codes(tokenizer, max(len(field.choices) for field in schema.values()))
    row_codes = []
    for field in schema.values():
        labels = rotated(codes[: len(field.choices)], code_offset)
        text = question_suffix(field, labels)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        choices = boundary_candidate_ids(tokenizer, text, labels)
        # Exactly the original prefix IDs followed by this branch's suffix IDs.
        row = prefix_ids + tokens
        rows.append(row)
        positions.append(len(row) - 1)
        candidates.append(choices)
        texts.append(text)
        slot_ids.append(tokens)
        row_codes.append(labels)
    if max(map(len, rows)) > max_input_tokens:
        raise ValueError("Request exceeds input token limit; no truncation performed")
    return CompiledSlots(
        rows, positions, candidates, texts, slot_ids, prefix, prefix_length, row_codes
    )
