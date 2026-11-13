"""The hybrid-thinking model's two modes, served by SGLang.

Implements the same surface as `hde.backend.TransformersScorer`:

    hidden_size                              the backbone's width
    score(state, schema)                     the fast mode, one RawFieldScores per question
    think(state, schema, names, budget)      the slow mode, for the named questions

so `SchemaDecisionEngine` takes either without knowing which.
"""

import json
import re
from dataclasses import dataclass, field, replace

import numpy as np

from hde.backends.slots import SLOT_FORMAT, boundary_candidate_ids, candidate_codes
from hde.core.pipeline_utils import RawFieldScores
from hde.core.prompts import build_content_messages, build_messages
from hde.core.prompts_utils import (
    ANSWER_BOUNDARY,
    CONTENT_FREE_INSTRUCTIONS,
    close_thinking,
    listable_options,
    open_thinking,
    question_body,
    shared_prefix,
    unlabelled_suffix,
)
from hde.readout.content_readout import OptionScore, option_token_ids

# `generate` takes `sampling_params` as a plain dict, so a key it does not know is
# dropped without a word -- an unseeded run reporting a seed. The name is therefore
# read off the installed SamplingParams rather than written down here.
#
# Where that class lives is the part this got wrong once. `sglang.sampling_params`
# was a guess, it does not exist in 0.5.9, and 27901 died on the
# ModuleNotFoundError. The first path below was measured there; the second is kept
# because an older layout is likelier than a new one nobody has seen. Both are
# tried and a failure becomes the same refusal as a missing name, because what the
# caller needs to know is that the run cannot be seeded, not which import failed.
SEED_MODULES = ("sglang.srt.sampling.sampling_params", "sglang.sampling_params")
SEED_NAMES = ("sampling_seed", "seed", "random_seed")
# Constrained decoding, read the same way and for the same reason. A proposal
# that comes back as free text has to be parsed, and parsing it is where four
# rounds of faults lived: nested boxes, unclosed thinking, a cut landing inside
# \text{}. A schema the engine enforces removes the parse rather than improving
# it.
#
# It does not stop an answer being an option letter. `{"answer": "B"}` satisfies
# this schema, and no pattern could rule it out either, because "2", "x" and "C"
# are all real MATH-500 answers. What keeps proposals index-free is the prompt,
# which mentions no options, and `looks_like_codes`, which stops a run that
# returns them anyway. The schema fixes the shape and nothing else.
SCHEMA_NAMES = ("json_schema", "response_format")
# Preferred over the schema, because JSON and LaTeX collide and a schema cannot
# separate them. `\\` is an escaped backslash to JSON and a line break to LaTeX,
# the same two characters, so no reader of a JSON string can be right about both
# `\\frac` (a model escaping properly) and `\begin{pmatrix}1\\2\end{pmatrix}`
# (a model writing maths). A regex fixes the same shape without an escaping layer
# under it: the answer is whatever sits between the quotes, read as written.
REGEX_NAMES = ("regex",)
PROPOSAL_REGEX = r'\{"answer": "[^"\n]{1,200}"\}'
PROPOSAL_SCHEMA = json.dumps({
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
})


# Everything JSON spells with a backslash that LaTeX also spells with one.
# `json.loads` is right about all of them and wrong for this: it reads
# "\frac{1}{2}" as formfeed + "rac{1}{2}", "\boxed{5}" as backspace +
# "oxed{5}", and \theta, \neq and \right the same way, without erroring,
# because each is a valid JSON escape. Every MATH-500 answer with LaTeX in it
# would have been quietly wrong.
ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"([^"\n]*)"')


def answer_of(text):
    r"""The answer out of a constrained proposal, exactly as the model wrote it.

    Read, not parsed, and deliberately not unescaped. `json.loads` is right about
    JSON and wrong for this: it reads `{"answer": "\frac{1}{2}"}` as formfeed
    followed by `rac{1}{2}`, and `\boxed`, `\theta`, `\neq` and `\right` the
    same way, without erroring, because each is a valid JSON escape and a LaTeX
    command. On MATH-500 every answer with maths in it would have been quietly
    wrong.

    Unescaping only `\\` and `\"` does not save it either, because `\\` is an
    escaped backslash to JSON and a line break to LaTeX -- the same two
    characters in `\begin{pmatrix}1\\2\end{pmatrix}`. There is no reader that is
    right about both, so the constraint is a regex with no escaping layer under
    it and this takes what is between the quotes.

    Refuses rather than falling back to the raw text: a response that is not the
    object asked for means the constraint did not apply, and taking the text
    anyway would put an unconstrained run inside a record that says otherwise.
    """
    found = ANSWER_FIELD.search(text)
    if not found:
        raise ValueError(
            f"a proposal came back without a JSON \"answer\" field, so the schema was "
            f"not enforced: {text[:200]!r}")
    return found.group(1)


@dataclass(frozen=True)
class SGLangSettings:
    """What the engine needs, and the two choices that are ours rather than its.

    `feature_layers` is the layer the decision head reads. An empty tuple turns
    hidden-state extraction off, which is cheaper and is the right setting when
    no head is being served.

    `batch_questions` sends a request's questions to the engine together, which
    is what the prefix cache rewards. SGLang has open issues where batched
    hidden states are attributed to the wrong request (sgl-project/sglang#8066,
    #4997), and that failure is silent: the head reads another question's state
    and the run completes. Set it false to serve questions one at a time, and
    run `inference.verify` on any engine version before trusting true.
    """

    model_id: str
    revision: str = "main"
    feature_layers: tuple[int, ...] = ()
    batch_questions: bool = False
    mem_fraction_static: float = 0.8
    max_input_tokens: int = 16384
    engine_kwargs: dict = field(default_factory=dict)


def readout_hidden_state(states, feature_layers=(-1,), position=-1):
    """The vector at the last prompt position, from what this engine returns.

    One parser, because there were two and they had diverged. This module read a
    three-dimensional return as `[position, layer, width]` and `inference.verify`
    read it as `[step, position, width]`. cdda372 settled it by measuring rather
    than by rank: a 15-token prompt gives `[1, 15, 5120]`, so the axis that moves
    with the prompt is the middle one and there is **no depth axis at all**.

    Under the wrong reading `feature_layers=(-1,)` was right by coincidence --
    the last of a position axis read as layers is the last position -- while
    `(48,)` silently returned the vector at position 48, or raised IndexError on
    a prompt shorter than 49 tokens. A head on any layer but the last therefore
    cannot be served here, and that is refused rather than approximated.
    """
    states = np.asarray(states, dtype=np.float32)
    if tuple(feature_layers) not in ((), (-1,)):
        raise RuntimeError(
            f"this engine returns one layer and no depth axis, so the head's "
            f"{list(feature_layers)} cannot be read from it: serve that head on "
            f"transformers, or train one on layer -1")
    # `position` is the row's last prompt position for the coded readout, and the last
    # *prefix* position for the content readout, which is not the row's end: a content
    # row continues past it with the option's own tokens.
    #
    # Negative, always, and that is the contract rather than a convenience. The engine
    # returns states for the tokens it computed, which under a prefix cache is the
    # tail of the row -- so an index counted from the start names a different token
    # depending on what was cached, and nothing in the reply says what was.
    #
    # **Settled, and it confirms this contract rather than excusing it.**
    # `hidden_positions` is recorded beside each row by `_hidden_with_shape` for
    # exactly this question, and 29701 answers it on both modes:
    #
    #     fast    n=1617   cached_tokens 0 on all
    #                      hidden_positions == prompt_tokens on all, median 227
    #     slow    n= 842   cached_tokens > 0 on all, median 1196
    #                      cached / prompt_tokens  median 0.9970, min 0.7160
    #                      hidden_positions  median 3, three distinct values
    #
    # **A cache hit ships only the recomputed tail**, and `hidden_positions` is
    # that tail exactly: `prompt_tokens - cached_tokens == hidden_positions` on
    # **842 of 842** slow readouts, with no exceptions. One asks about a 1,199-token
    # prompt and gets 3 positions. So an index counted from the start would name a
    # different token depending on what was cached, which is the contract above,
    # now measured instead of argued.
    #
    # 836 of the 842 re-prefill only the 3 or 4 boundary tokens. The six that do
    # not are **one item**, `hard-sol-c-multi_hop-07`, identical in all six
    # escalating policies at cached 2594, prompt 3623, tail 1029 -- so it is
    # deterministic and a property of that item, not cache pressure, which rules
    # out the eviction story the numbers first suggest. 1029 is 1024 + 5, the full
    # budget's generation plus the boundary, so for that one item the generated
    # continuation was not retained and the follow-up re-read all of it. What makes
    # it different is not in these fields.
    #
    # **A hidden-state request can read the cache.** `think`'s follow-up (`:811`)
    # passes no `want_hidden` and so defaults to True, and it gets 99.7% of its
    # prompt from cache. So asking for states does not disable reuse.
    #
    # What is not explained by anything in these fields is the fast pass's zero.
    # `bench_policies` iterates batches outer and policies inner (`:179`, `:182`),
    # so the seven requests on one prompt are **adjacent**, identical in text, and
    # every one of them computes the whole prompt. Neither request order nor
    # eviction can account for that, and both were proposed and withdrawn here.
    # The candidate left, stated as a candidate: a request that asks for states
    # leaves no reusable entry, so a later identical one has nothing to hit, while
    # it can still read an entry left by a `want_hidden=False` call -- which is
    # exactly the asymmetry between the fast pass and the follow-up, since the
    # follow-up's prefix was written by the generation call at `:801`.
    #
    # That candidate is weaker than it looks, and the reason is in this module's
    # own prompts. `close_thinking` and `open_thinking` share `shared_prefix` and
    # then diverge, the fast path adding about five tokens and the slow path one,
    # so a fast request is a sibling branch off the shared prefix rather than a
    # prefix of the stored path. Its median is 227 tokens, nearly all of them
    # `shared_prefix`, which the generation call computed one or two requests
    # earlier on the same item. A radix tree would be expected to match at the
    # divergence and report that length. It reports zero. Serving a shared prefix
    # is what such a cache is for, so the zero is more surprising than any story
    # offered here, and no fourth one is offered.
    #
    # Two fields would settle it and neither exists: the generation call's own
    # `cached_tokens`, and the shared-prefix length, so that a zero can be read as
    # "nothing was shareable" rather than "something was and nothing was served".
    # Their absence is why this has been guessed at three times.
    #
    # Two wrong explanations were committed here before this one. The first said the
    # cache never hits when states are requested, from the `fast` records alone; the
    # 842 `slow` records in the same file refute it. The second said the fast zero
    # was eviction across 231 intervening items; the loop order refutes it. exps
    # caught both, and the identity above is what survives of all of it.
    #
    # **This is about requests that ask for states, and the escalation is not one.**
    # `think`'s generation call passes `want_hidden=False`, and its prompt is a
    # strict prefix of the fast pass's -- `open_thinking` adds `\n` where
    # `close_thinking` adds `\n\n</think>\n\n`, so the fast prompt extends the slow
    # one by three tokens (9122c1d). Nothing there forces recomputation, so an
    # escalation continues from what the fast pass computed, which is the design.
    if position >= 0:
        raise ValueError(
            f"position {position} counts from the start of the row, and the engine "
            f"returns states only for the tokens it computed: under a prefix cache "
            f"that is the tail, so the index would name a different token depending "
            f"on what was cached. Count from the end")
    # The rank first: a shape nobody has seen is a different question from a reply
    # that is too short, and the message for it names the probe that identifies it.
    if states.ndim not in (2, 3):
        raise RuntimeError(f"unexpected hidden_states shape {states.shape}; "
                           f"scripts/probe_sglang.py identifies it by what it varies with")
    positions = states.shape[-2]
    if positions < -position:
        raise RuntimeError(
            f"the engine returned {positions} positions and the readout is {-position} "
            f"from the end, so the state asked for was not computed in this request")
    if states.ndim == 3:                     # [step, position, width], as measured
        return states[0, position]
    return states[position]                  # [position, width]; not seen on this build


class SGLangScorer:
    """Fast and slow modes over one SGLang engine."""

    def __init__(self, engine, tokenizer, settings, hidden_size):
        # Also here, for every way a scorer is built that is not `load`: the engine
        # is passed in by `SGLangServerScorer`, by `verify`, and by the tests.
        self._check_layers(settings)
        self.engine = engine
        self.tokenizer = tokenizer
        self.settings = settings
        self._hidden_size = hidden_size

    @staticmethod
    def _hidden_size(model_id, revision):
        """The backbone's width, read from the checkpoint rather than asked of transformers.

        SGLang pins a transformers that does not recognise every architecture it can
        itself serve, so `AutoConfig` refuses checkpoints the engine loads happily.
        The width is a number in config.json; reading it needs nobody's opinion of
        the architecture.
        """
        import json
        from pathlib import Path

        path = Path(model_id) / "config.json"
        if path.exists():
            config = json.loads(path.read_text())
        else:
            from huggingface_hub import hf_hub_download

            config = json.loads(Path(hf_hub_download(
                model_id, "config.json", revision=revision)).read_text())
        for node in (config.get("text_config", {}), config):
            if "hidden_size" in node:
                return int(node["hidden_size"])
        raise ValueError(f"no hidden_size in the config of {model_id}")

    @staticmethod
    def _check_layers(settings):
        """Refuse a head this engine cannot serve, before anything has been loaded.

        It was in `__init__` alone, which `load` reaches only after
        `sgl.Engine(...)` has loaded the backbone -- so on the real path the
        refusal arrived after the 27B load it exists to save, which is precisely
        the cost 28170 paid. code-reviewer caught that the comment and the test
        name both claimed an ordering the code did not have.
        """
        if tuple(settings.feature_layers) not in ((), (-1,)):
            raise ValueError(
                f"this engine returns only the last hidden layer, so a head reading "
                f"{list(settings.feature_layers)} cannot be served through it. Train one "
                f"with hde-train-head --candidate layer_-1, or serve it on transformers")

    @classmethod
    def load(cls, settings):
        # First, before `import sglang` and before the config is read, so that a
        # head this engine cannot serve costs nothing to refuse.
        cls._check_layers(settings)
        import sglang as sgl
        from transformers import AutoTokenizer

        hidden_size = cls._hidden_size(settings.model_id, settings.revision)
        engine = sgl.Engine(
            model_path=settings.model_id,
            revision=settings.revision,
            mem_fraction_static=settings.mem_fraction_static,
            # Extraction has to be enabled when the engine starts; it cannot be
            # asked for per request afterwards.
            enable_return_hidden_states=bool(settings.feature_layers),
            **settings.engine_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(settings.model_id, revision=settings.revision)
        return cls(engine, tokenizer, settings, hidden_size)

    @property
    def hidden_size(self):
        return self._hidden_size

    def _codes_for(self, schema):
        width = max(len(field.choices) for field in schema.values())
        return candidate_codes(self.tokenizer, width)

    def _render(self, state, schema, names=None, thinking=False, mode=None):
        """One render per question; where it is cut picks the mode.

        Three cuts, from the one render. ``fast`` closes the thinking block and
        appends the answer boundary, so the answer is the next thing the model
        would produce and nothing is generated. ``slow`` leaves the block open and
        reasons. ``short`` closes the block and stops there, so the model writes
        its answer without a reasoning trace -- an empty ``<think></think>`` is how
        ThinkSwitcher elicits its short mode, and this is that prompt.

        `thinking` remains as the two-mode spelling its callers use.

        Both modes come from the same string, so an escalation continues from the
        fast pass's cache instead of prefilling the prompt again. Questions are
        still isolated from one another, as the schema requires; what they share
        is now the system turn and the state, which the engine's prefix cache
        matches on just as it matched the old explicit prefix.
        """
        mode = mode if mode is not None else ("slow" if thinking else "fast")
        if mode not in ("fast", "slow", "short"):
            raise ValueError(f"mode must be fast, slow or short; got {mode!r}")
        codes = self._codes_for(schema)
        prompts, labels = [], []
        for name, question in schema.items():
            if names is not None and name not in names:
                continue
            own = codes[: len(question.choices)]
            rendered = self.tokenizer.apply_chat_template(
                build_messages(state, question_body(question, own)),
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            shared = shared_prefix(rendered)
            # The slow mode gets the render as it stands, block open, preamble and
            # all; the fast mode closes the block and reads at the boundary. Both
            # extend `shared` token for token, so the engine's prefix cache serves
            # an escalation from what the fast pass already computed.
            if mode == "slow":
                prompts.append(open_thinking(shared))
            elif mode == "short":
                prompts.append(close_thinking(shared))
            else:
                prompts.append(close_thinking(shared) + ANSWER_BOUNDARY)
            labels.append((name, own))
        return prompts, labels

    def _candidate_ids(self, codes):
        """The ids `compile_prefix_slots` reads, from the same function it reads them.

        This used to encode the bare code, giving the id of `"A"` where the
        transformers path reads `" A"` -- two different rows of the same logit
        vector, both single tokens, nothing failing. Over MMLU-Pro's stride-8 1500
        on the 27B the two backends' coded readouts agreed on 78.1% of questions
        with systematic shifts, which is what sent code-reviewer looking.

        The anchor is the answer boundary, which is where the code is read on both
        paths and what `candidate_codes` validated the code set against.
        """
        return boundary_candidate_ids(self.tokenizer, ANSWER_BOUNDARY, codes)

    def _generate(self, prompts, max_new_tokens, candidate_ids_per_prompt, want_hidden=None):
        """Logprobs for the candidate codes, and the hidden state beside them.

        `want_hidden` is False on the slow mode's generation call, where nothing reads
        the states: the head reads the fast pass, and `think` reads its own follow-up.
        Asking for them anyway makes the engine return a state per generated token, and
        this repository has already measured what that costs -- 48 MB of JSON for one
        900-token state, 14.3 s against 0.7 s, in `experiments/browsergym/README.md`.
        On JevBench it showed up as a slow path 1.4x slower with a head than without
        one (P95 37.5 s against 23.3 s) on the same generation, which exps found and
        could not explain.
        """
        if want_hidden is None:
            want_hidden = bool(self.settings.feature_layers)
        want_hidden = want_hidden and bool(self.settings.feature_layers)
        results = []
        groups = ([list(range(len(prompts)))] if self.settings.batch_questions
                  else [[i] for i in range(len(prompts))])
        for group in groups:
            outputs = self.engine.generate(
                [prompts[i] for i in group],
                sampling_params=[{"max_new_tokens": max_new_tokens, "temperature": 0.0}
                                 for _ in group],
                return_logprob=True,
                token_ids_logprob=[candidate_ids_per_prompt[i] for i in group],
                return_hidden_states=want_hidden,
            )
            outputs = outputs if isinstance(outputs, list) else [outputs]
            results.extend(zip(group, outputs))
        return [output for _, output in sorted(results, key=lambda pair: pair[0])]

    @staticmethod
    def _parameter(names, what):
        """The first of `names` this SGLang's SamplingParams takes.

        `generate` takes `sampling_params` as a plain dict, so a key it does not
        know is dropped without a word: a run that reports a seed it never used,
        or a schema it never enforced. So the name is read off the installed class
        rather than written down, and nothing found is one `ValueError` covering a
        missing class and a missing parameter alike -- the caller's question is
        whether the run can have the thing, not which lookup failed.
        """
        import importlib
        import inspect

        tried = {}
        for path in SEED_MODULES:
            try:
                params = importlib.import_module(path).SamplingParams
            except (ImportError, AttributeError) as failure:
                tried[path] = f"{type(failure).__name__}: {failure}"
                continue
            accepted = inspect.signature(params.__init__).parameters
            for name in names:
                if name in accepted:
                    return name
            tried[path] = f"takes none of {names}; it takes {sorted(accepted)}"
        raise ValueError(
            f"this SGLang has no {what} this knows how to reach. Run without it and say "
            f"so in the record, or add the name or path below once you have read it off "
            f"the installed package: {tried}")

    @classmethod
    def seed_parameter(cls):
        """What this SGLang calls a per-request sampling seed."""
        return cls._parameter(SEED_NAMES, "per-request sampling seed")

    @classmethod
    def schema_parameter(cls):
        """What this SGLang calls a per-request JSON schema."""
        return cls._parameter(SCHEMA_NAMES, "per-request JSON schema")

    _constraint = None

    @classmethod
    def proposal_constraint(cls):
        r"""`(kind, parameter)` for holding a proposal to its shape, resolved once.

        The regex is preferred and the schema is a fallback that changes answers
        rather than merely constraining them: `\sqrt`, `\pi` and `\cdot` are not
        legal JSON escapes, so a JSON grammar forbids the very answers MATH-500 is
        made of, and a run under it is measuring something else. A warning is not
        enough for that, so the kind goes in the record and a reader can tell the
        two apart afterwards.
        """
        if cls._constraint is None:
            try:
                cls._constraint = ("regex", cls._parameter(REGEX_NAMES, "regex constraint"))
            except ValueError:
                cls._constraint = ("json_schema", cls.schema_parameter())
                print("warning: this SGLang has no regex constraint, so proposals are "
                      "schema-constrained. A JSON grammar forbids \\sqrt, \\pi and "
                      "\\cdot, so some answers cannot be written at all", flush=True)
        return cls._constraint

    def propose(self, state, instructions, width=10, max_new_tokens=32, temperature=0.7,
                seed=None, verified=True):
        """Answers this question could have, for a question that arrived with none.

        SGLang has no beam search, so the proposals are `width` independent
        samples rather than a ranked beam. Nothing is lost that the decision head
        reads: it reads the readout's distribution over the constructed codes,
        which is a later forward pass either way. What changes is the ranking,
        and sampling supplies one a beam cannot — how many of the samples agree
        on an answer (`hde.construct.FREQUENCY`) as well as how likely each
        one is (`LIKELIHOOD`, the summed token logprobs returned here).

        Duplicates are returned rather than collapsed, since the repeats are the
        frequency signal. At temperature zero every sample is the same one and
        the construction degenerates, which `construct` detects and marks; the
        temperature that produced a coverage number has to be reported with it.

        Thinking stays switched off, so these are the fast mode's own answers.
        The slow mode renders its own prompt and never receives them.

        `verified` has the engine enforce a JSON schema on what comes back, so a
        proposal arrives as `{"answer": ...}` and is read rather than parsed. The
        parsing it replaces is where four rounds of faults lived -- a nested
        `\boxed{}`, thinking that was not stripped, a cut landing inside
        `\text{}` -- and each was found by reading responses rather than by a
        test. It does not stop an answer being an option letter: the prompt and
        `looks_like_codes` do that. Off only for measuring what it changes, and
        `proposal_constraint` says which was used.

        `seed` is passed to the engine as a per-request sampling seed. Two runs at
        one configuration came out at 45% and 35% coverage with nothing to separate
        them, which is what an unseeded sampler buys. It is a seed and not a
        guarantee: SGLang's batching makes the arithmetic depend on what else is in
        the batch, so identical seeds can still diverge unless the engine was
        started for deterministic inference. A seeded number is reproducible on the
        same engine settings and reports which draw it came from either way.
        """
        from hde.core.prompts import build_proposal_messages

        if width < 2:
            raise ValueError("fewer than two proposals is not a choice")
        if temperature <= 0:
            raise ValueError("sampling at temperature zero returns one answer width times")
        prompt = self.tokenizer.apply_chat_template(
            build_proposal_messages(state, instructions, verified=verified),
            tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        sampling = {"max_new_tokens": max_new_tokens, "temperature": temperature,
                    "n": width}
        if seed is not None:
            sampling[self.seed_parameter()] = int(seed)
        if verified:
            kind, name = self.proposal_constraint()
            sampling[name] = PROPOSAL_REGEX if kind == "regex" else PROPOSAL_SCHEMA
        output = self.engine.generate(prompt, sampling_params=sampling, return_logprob=True)
        samples = list(self._samples_from(output))
        return [(answer_of(text) if verified else text, score) for text, score in samples]

    def solve(self, state, question, budget, answer_tokens=48):
        """Think about a question with no options, then write the answer out.

        `think` renders its prompt through `question_body`, which writes every
        option into it. When the options were constructed by the fast mode that
        turns the slow mode into a policy answering from the fast mode's
        proposals, which is not the policy the always-thinking column measures.
        This renders the question alone.

        A budget that runs out mid-thought is made to answer anyway, the way
        `think` does: the close tag is appended and the model writes its answer
        after it. Without that the two slow columns are different policies --
        one always answers at the cap and the other sometimes returns a trace --
        and greedy decoding reaches the cap often, because it loops.

        Returns `(text, thinking_tokens, truncated)`.
        """
        from hde.core.prompts import build_open_thinking_messages

        prompt = self.tokenizer.apply_chat_template(
            build_open_thinking_messages(state, question), tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        written, spent = self._complete(prompt, budget)
        if "</think>" in written:
            return written, spent, False
        forced, extra = self._complete(prompt + written + "</think>\nAnswer:", answer_tokens)
        return written + "</think>\nAnswer:" + forced, spent + extra, True

    def _complete(self, prompt, max_new_tokens):
        output = self.engine.generate(
            prompt, sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0.0})
        row = output[0] if isinstance(output, list) else output
        if not isinstance(row, dict):
            raise RuntimeError(f"the server returned {type(row).__name__}, not a completion")
        meta = row.get("meta_info", {})
        text = row.get("text", "")
        spent = meta.get("completion_tokens")
        if spent is None:
            # Counting it zero would put a free slow column in the table.
            raise RuntimeError("the server reported no completion_tokens, so the slow "
                               "column's cost cannot be measured")
        return text, int(spent)

    @staticmethod
    def _samples_from(output):
        """(text, summed output logprob) per sample, however the server packs them.

        One request with `n` set comes back as a list when the engine batches and
        as a single object when it does not, and the logprobs sit under different
        keys depending on the version, so this reads what is there and says so
        when nothing is.
        """
        rows = output if isinstance(output, list) else [output]
        samples = []
        for row in rows:
            meta = row.get("meta_info", {}) if isinstance(row, dict) else {}
            text = row.get("text", "") if isinstance(row, dict) else str(row)
            entries = meta.get("output_token_logprobs") or []
            if not entries:
                raise RuntimeError(
                    "the server returned samples without logprobs, so ranking them by "
                    "likelihood would rank them by arrival order. Ask for them with "
                    "return_logprob, or rank by frequency instead.")
            score = float(sum(entry[0] for entry in entries if entry and entry[0] is not None))
            samples.append((text, score))
        if not samples:
            raise RuntimeError("the server returned no samples to propose from")
        return samples

    @staticmethod
    def _logits_from(output, candidate_ids):
        """The candidate logprobs at the readout position, ordered as the codes are.

        The engine reports these under `output_token_ids_logprobs`, one entry per
        generated position, each a list of (logprob, token_id, text) for the ids
        that were asked for. With one generated token there is one entry, and it is
        the distribution the boundary set up, which is what the fast mode reads.

        `input_token_ids_logprobs` exists alongside it and is `[None]` unless
        `logprob_start_len` was set. Reading that one instead is how this crashed
        on the first step of the first episode, so the entries are searched for the
        first that holds anything rather than indexed by position.

        Order is by token id, not by the position the engine returned them in.
        """
        meta = output["meta_info"]
        rows = [row for row in (meta.get("output_token_ids_logprobs") or []) if row]
        if not rows:
            rows = [row for row in (meta.get("input_token_ids_logprobs") or []) if row]
        if not rows:
            raise RuntimeError(
                "SGLang returned no token-id logprobs. Both output_token_ids_logprobs "
                "and input_token_ids_logprobs were empty or all None; was "
                "token_ids_logprob passed, and was at least one token generated?")
        table = {int(token): float(value) for value, token, *_ in rows[0]}
        missing = [t for t in candidate_ids if t not in table]
        if missing:
            raise RuntimeError(f"SGLang omitted logprobs for token ids {missing}")
        return np.array([table[t] for t in candidate_ids], dtype=np.float32)

    def _hidden_from(self, output):
        """The readout position's vector, stacked as the head expects."""
        return self._hidden_with_shape(output)[0]

    def _hidden_with_shape(self, output):
        """The readout vector, and how many positions the engine returned for it.

        The position count is the one number that settles what the reply costs
        and whether the prefix cache shortened it: the head reads one position
        and the engine returns some number of them, and nothing recorded which.
        It is free here -- `readout_hidden_state` already materialises the array.
        """
        if not self.settings.feature_layers:
            return None, None
        states = np.asarray(output["meta_info"]["hidden_states"], dtype=np.float32)
        return (readout_hidden_state(states, self.settings.feature_layers)[None, :],
                int(states.shape[-2]))

    def _fields(self, prompts, labels, outputs, generated):
        fields = []
        for (name, codes), prompt, output in zip(labels, prompts, outputs):
            hidden, positions = self._hidden_with_shape(output)
            meta = output["meta_info"]
            fields.append(RawFieldScores(
                name=name,
                logits=self._logits_from(output, self._candidate_ids(codes)),
                input_tokens=int(meta.get("prompt_tokens", 0)),
                hidden=hidden,
                generated_tokens=int(meta.get("completion_tokens", 0)) if generated
                else 0,
                provenance={
                    "module": "sglang",
                    "slot_format": SLOT_FORMAT,
                    "boundary": ANSWER_BOUNDARY.strip(),
                    "codes": list(codes),
                    "batched": self.settings.batch_questions,
                    # **What the engine reused and what it sent back.** Both modes
                    # are cut from one render so an escalation re-sends a prompt
                    # the fast pass computed, and whether the engine serves it
                    # from its radix cache was an assertion in this file's
                    # docstrings and an inference from wall-clock. `cached_tokens`
                    # is the engine's own count of prompt tokens it did not
                    # recompute; `hidden_positions` is how many states came back
                    # where the head reads one. Absent keys stay None rather than
                    # zero, since this SGLang may not report them.
                    "cached_tokens": meta.get("cached_tokens"),
                    # The denominator, from the same reply: `641 cached` is a
                    # count until it is `641 of 648`. `input_tokens` above holds
                    # the same number, but the bench reads provenance and not the
                    # field, so it arrived as null where it was needed.
                    "prompt_tokens": meta.get("prompt_tokens"),
                    "hidden_positions": positions,
                },
            ))
        return tuple(fields)

    def score(self, state, schema):
        """The fast mode: no generated token, the answer read at the boundary."""
        prompts, labels = self._render(state, schema, thinking=False)
        candidate_ids = [self._candidate_ids(codes) for _, codes in labels]
        outputs = self._generate(prompts, max_new_tokens=1, candidate_ids_per_prompt=candidate_ids)
        return self._fields(prompts, labels, outputs, generated=False)

    def score_options(self, state, question, content_free=True):
        """Score each option by its own text, with no label between the two.

        The same readout as `TransformersScorer.score_options`, returning the same
        `(scores, rewritten)`, so the engine does not know which backend is under
        it. Nothing is sampled: a row is the prefix followed by the option's own
        tokens, and the engine is asked what it gave tokens it was handed rather
        than what it would have chosen.

        The scores will not equal the transformers ones and are not interchangeable
        with them. The two implementations' readout states agree to cosine 0.9995
        and differ by about 3% in magnitude, and a score here is a sum over several
        such positions, so a table may not mix the backends without saying which
        produced each row.
        """
        scored, rewritten, hidden = self._score_options(state, question)
        if not content_free:
            return scored, rewritten, hidden
        blank = question.model_copy(update={"instructions": CONTENT_FREE_INSTRUCTIONS})
        # The prior pass asks a question with nothing in it, so its readout state is
        # not this question's and is discarded rather than returned.
        priors, _, _ = self._score_options(state, blank)
        return [replace(score, prior=prior.total)
                for score, prior in zip(scored, priors, strict=True)], rewritten, hidden

    def _score_options(self, state, question):
        prefix = self.tokenizer.apply_chat_template(
            build_content_messages(state), tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        ) + unlabelled_suffix(question)
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        # The same rewriting the listing does, from the same function, because the
        # two disagreeing about an option's text is the fault being avoided.
        options, rewritten = listable_options(question)
        # The separating space is part of the option, not of the prompt: BPE binds
        # it to the following word, so a prompt ending in one cannot be a prefix of
        # the joint encoding.
        per_option = [option_token_ids(self.tokenizer, prefix, " " + text)
                      for text in options]
        outputs = self._teacher_force([prefix_ids + ids for ids in per_option],
                                      len(prefix_ids))
        scores = [
            OptionScore(index=index, text=text, tokens=len(ids),
                        total=self._option_total(output, ids))
            for index, (text, ids, output)
            in enumerate(zip(options, per_option, outputs, strict=True))
        ]
        # The prefix is identical in every row and attention is causal, so the state at
        # its last position is the same in all of them and is read from the first.
        #
        # Counted from the end of the row, not from the start of the prompt. The engine
        # returns states only for the tokens it computed, and the prefix cache means
        # that is the tail: keys and values are cached, the residual stream is not. An
        # absolute `len(prefix) - 1` therefore read another token's state as soon as
        # anything shared that prefix -- which is every row after the first, and every
        # row at all once another question has used the same chat template. The same
        # move as `_option_total`'s tail slice, and for the same reason.
        hidden = None
        if self.settings.feature_layers:
            states = outputs[0]["meta_info"].get("hidden_states")
            if states is None:
                raise RuntimeError(
                    "this engine returned no hidden states for the content readout; it "
                    "must be started with enable_return_hidden_states when a head is to "
                    "be trained or served under it")
            hidden = readout_hidden_state(states, self.settings.feature_layers,
                                          position=-(len(per_option[0]) + 1))[None, :]
        return scores, rewritten, hidden

    def _teacher_force(self, rows, prefix_length):
        """One reply per row, carrying the log-probability of each given token.

        Sent as token ids, not as text. Re-tokenising `prefix + option` on the far
        side could merge across the boundary that `option_token_ids` has just
        proved separable, and then the tokens whose log-probabilities come back
        are not the tokens whose count the normalisations divide by.

        Grouped by `batch_questions` for the reason that setting exists. SGLang's
        open misattribution bugs are about what a request gets back when it shares
        a forward pass, and here an option is a request exactly as a question is.
        The rows share their whole prefix, so the engine's prefix cache pays for
        running them one at a time.
        """
        groups = ([list(range(len(rows)))] if self.settings.batch_questions
                  else [[index] for index in range(len(rows))])
        results = []
        for group in groups:
            outputs = self.engine.generate(
                input_ids=[rows[index] for index in group],
                sampling_params=[{"max_new_tokens": 1, "temperature": 0.0}
                                 for _ in group],
                return_logprob=True,
                return_hidden_states=bool(self.settings.feature_layers),
                # One position before the first option token: the entry for a token
                # is the distribution it was drawn from, which sits at the position
                # before it. Asking from here rather than from `prefix_length` keeps
                # the request right under either reading of the parameter, and
                # `_option_total` checks the ids rather than trusting the offset.
                logprob_start_len=[prefix_length - 1] * len(group),
            )
            outputs = outputs if isinstance(outputs, list) else [outputs]
            results.extend(zip(group, outputs))
        return [output for _, output in sorted(results, key=lambda pair: pair[0])]

    @staticmethod
    def _option_total(output, option_ids):
        """The summed log-probability of this option's own tokens.

        Found by the ids that came back, not by an offset into the reply.
        `logprob_start_len` is the one part of this whose meaning has moved between
        SGLang versions, and an off-by-one there scores the last prefix token
        instead of the option's last: a number that looks ordinary, is wrong by one
        term in every option, and leaves the ranking intact so that nothing
        downstream reports it.
        """
        meta = output["meta_info"]
        entries = [entry for entry in (meta.get("input_token_logprobs") or []) if entry]
        if not entries:
            raise RuntimeError(
                "SGLang returned no input_token_logprobs, so the option's own tokens "
                "were never scored. Was return_logprob passed with logprob_start_len?")
        if len(entries) < len(option_ids):
            raise RuntimeError(
                f"SGLang returned {len(entries)} input log-probabilities for an option "
                f"of {len(option_ids)} tokens, so the reply begins after the option "
                f"does and the first of its tokens is missing")
        tail = entries[-len(option_ids):]
        got = [int(entry[1]) for entry in tail]
        if got != list(option_ids):
            raise RuntimeError(
                f"the log-probabilities came back for token ids {got}, not this "
                f"option's {list(option_ids)}: the reply is not aligned with the "
                f"request, so the sum would be over the wrong positions")
        if any(entry[0] is None for entry in tail):
            raise RuntimeError(
                "SGLang returned None for one of the option's tokens, which is what it "
                "returns for the first token of a sequence; the prefix is missing")
        return float(sum(float(entry[0]) for entry in tail))

    def short_cot(self, state, schema, names, budget):
        """ThinkSwitcher's short mode: an empty thinking block, then an answer.

        `think` with one difference, the cut: the block is closed and empty before
        the model writes, so it answers without a reasoning trace but still decodes
        autoregressively. That is the distinction the fast mode does not have --
        the fast mode generates nothing at all -- and it is what makes this a
        length baseline rather than a second copy of ours.

        Everything after the prompt is `think`'s: the same generation call, the
        same answer boundary appended to what was written, the same candidate
        readout at it. A separate readout here is how the two modes would come to
        disagree for a reason that is not the prompt.
        """
        return self._generate_then_read(state, schema, names, budget, mode="short")

    def think(self, state, schema, names, budget):
        """The slow mode: reason up to `budget` tokens, then read the same candidates."""
        return self._generate_then_read(state, schema, names, budget, mode="slow")

    def _generate_then_read(self, state, schema, names, budget, mode):
        """Generate under `mode`, then read the candidates at the answer boundary."""
        prompts, labels = self._render(state, schema, names=set(names), mode=mode)
        candidate_ids = [self._candidate_ids(codes) for _, codes in labels]
        # No hidden states for the reasoning itself: only its text and token count are
        # read, and the state the head needs comes from the follow-up read below.
        reasoned = self._generate(prompts, max_new_tokens=budget,
                                 candidate_ids_per_prompt=candidate_ids,
                                 want_hidden=False)
        # The answer is read after the reasoning, at the same boundary, so the
        # typed answer is assembled in code and nothing is parsed from the text.
        followups, spent, traces = [], [], []
        for prompt, output in zip(prompts, reasoned):
            followups.append(prompt + output["text"] + ANSWER_BOUNDARY)
            spent.append(int(output["meta_info"].get("completion_tokens", 0)))
            traces.append(output["text"])
        outputs = self._generate(followups, max_new_tokens=1,
                                 candidate_ids_per_prompt=candidate_ids)
        fields = self._fields(followups, labels, outputs, generated=False)
        return tuple(
            RawFieldScores(name=f.name, logits=f.logits, input_tokens=f.input_tokens,
                           hidden=f.hidden, generated_tokens=tokens, provenance=f.provenance,
                           reasoning=trace)
            for f, tokens, trace in zip(fields, spent, traces)
        )
