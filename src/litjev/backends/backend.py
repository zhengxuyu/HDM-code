"""Two forwards: shared prefix prefill, then independent cached answer branches.

The optional slow path re-runs escalated branches with the backbone's own thinking
mode and reads the same answer boundary again. The backbone is never modified.
"""

import threading
from dataclasses import dataclass, replace

import torch
from transformers.generation import StoppingCriteria, StoppingCriteriaList

from litjev.backends.slots import (
    SLOT_FORMAT,
    boundary_candidate_ids,
    candidate_codes,
    compile_prefix_slots,
    compile_slots,
)
from litjev.core.pipeline_utils import RawFieldScores
from litjev.core.prompts import (
    build_content_messages,
    build_decision_messages,
    build_messages,
    build_thinking_messages,
)
from litjev.core.prompts_utils import (
    ANSWER_BOUNDARY,
    CONTENT_FREE_INSTRUCTIONS,
    close_thinking,
    listable_options,
    open_thinking,
    question_body,
    shared_prefix,
    unlabelled_suffix,
)
from litjev.readout.content_readout import OptionScore, option_token_ids, summed_logprob
from litjev.vision import VisualState, validate_image

THINK_TOKENS = ("<think>", "</think>")


@dataclass(frozen=True)
class ModelSettings:
    model_id: str = "Qwen/Qwen3.8-27B"
    revision: str = "main"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    max_input_tokens: int = 16384
    feature_layers: tuple[int, ...] = ()
    # Rotates the answer codes against the option slots, so that a readout
    # preferring a letter and one preferring a slot stop giving the same answer.
    # Zero is every run to date; anything else is a run asking what it reads.
    code_offset: int = 0
    # Render both modes from one string, the way `inference.sglang_backend` does,
    # instead of from this path's own two renders.
    #
    # **Off by default, because it changes what the fast pass reads.** The two
    # renders differ in the system turn and in the thinking block, so the hidden
    # state a head is fitted to is taken at a different prompt. Switching the
    # default would quietly make `records-4b-27063` -- and every transformers
    # number built on it, `tab:features` among them -- incomparable with whatever
    # ran next. A run asks for this, and its records say so in their provenance.
    #
    # It exists because the layer ablation cannot otherwise be measured on the
    # prompt the served heads actually see: the unified render lives on the
    # SGLang path, and that engine returns only the final hidden layer (its
    # states come back as [step, position, width], with no layer axis), so a
    # sweep over layers has to run here. The helpers called below are the same
    # ones that path calls, so the two renders are one definition.
    unified_prompt: bool = False


class SuffixStop(StoppingCriteria):
    """Stop a row once its generated tail equals the closing delimiter."""

    def __init__(self, suffix, prompt_width):
        self.suffix = suffix
        self.prompt_width = prompt_width

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[:, self.prompt_width :]
        if generated.shape[1] < len(self.suffix):
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        tail = torch.tensor(self.suffix, device=input_ids.device)
        return (generated[:, -len(self.suffix) :] == tail).all(dim=1)


class TransformersScorer:
    def __init__(
        self,
        model,
        tokenizer,
        max_input_tokens=16384,
        processor=None,
        feature_layers=(),
        think_tokens=THINK_TOKENS,
        code_offset=0,
        unified_prompt=False,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.lock = threading.Lock()
        self.processor = processor
        self.feature_layers = tuple(feature_layers)
        self.think_tokens = think_tokens
        # Rotates the letters against the slots. A run that sets it is asking what
        # the readout is reading, not fixing it, and the answer belongs beside the
        # number that run produces.
        self.code_offset = code_offset
        self.unified_prompt = unified_prompt

    @classmethod
    def load(cls, settings: ModelSettings):
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        config = AutoConfig.from_pretrained(settings.model_id, revision=settings.revision)
        loader = (
            AutoModelForImageTextToText if config.model_type == "qwen3_5" else AutoModelForCausalLM
        )
        model = loader.from_pretrained(
            settings.model_id,
            revision=settings.revision,
            dtype=getattr(torch, settings.dtype),
            device_map=settings.device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(settings.model_id, revision=settings.revision)
        processor = (
            AutoProcessor.from_pretrained(settings.model_id, revision=settings.revision)
            if config.model_type == "qwen3_5"
            else None
        )
        return cls(model, tokenizer, settings.max_input_tokens, processor,
                   settings.feature_layers, code_offset=settings.code_offset,
                   unified_prompt=settings.unified_prompt)

    @property
    def hidden_size(self):
        config = getattr(self.model.config, "text_config", self.model.config)
        return config.hidden_size

    def score(self, state, schema):
        with self.lock, torch.inference_mode():
            if self.unified_prompt:
                return self._score_unified(state, schema)
            return self._score(state, schema)

    def score_options(self, state, question, content_free=True):
        """Score each option by its own text, with no label between the two.

        One batched prefill of k rows sharing the prefix, teacher-forced, nothing
        sampled. Returns `(scores, rewritten)`: an `OptionScore` per option
        carrying the summed log-probability, the token count the normalisations
        divide by, and -- when `content_free` -- the same option's score after a
        question with nothing in it to answer, which is what PMI subtracts; and
        how many of this question's options had to be put on one line.

        The count is returned rather than left on the scorer. As instance state
        it was overwritten by every later call, including by the content-free
        pass two lines down, so what a caller read afterwards was whatever the
        last question happened to need.

        Returns `(scores, rewritten, hidden)`. `hidden` is the state at the readout
        position for the layers this scorer was built with, or `None` when it was
        built with none -- without it a head cannot be trained or served under this
        readout, which is what kept routing a coded-only capability.
        """
        with self.lock, torch.inference_mode():
            scored, rewritten, hidden = self._score_options(state, question)
            if not content_free:
                return scored, rewritten, hidden
            blank = question.model_copy(
                update={"instructions": CONTENT_FREE_INSTRUCTIONS})
            # The prior pass asks a question with nothing in it, so its readout state
            # is not this question's and is discarded rather than returned.
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
        per_option = [option_token_ids(self.tokenizer, prefix, " " + text) for text in options]

        device = self.model.get_input_embeddings().weight.device
        rows = [prefix_ids + ids for ids in per_option]
        width = max(len(row) for row in rows)
        pad = self._pad_id()
        # Left-padding would move the prefix, and the prefix is where every
        # option's first token is predicted from.
        input_ids = torch.tensor([row + [pad] * (width - len(row)) for row in rows],
                                 device=device)
        attention = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows],
                                 device=device)
        # Only the positions `summed_logprob` reads. Three tensors of
        # `batch x width x vocab` used to be alive here at once -- the bf16 logits
        # still held by `output`, the float32 copy `.float()` makes, and
        # log_softmax's float32 result -- which for ten options over a 1000-token
        # prefix on the 27B is 5.6 + 11.3 + 11.3 = 28 GiB.
        #
        # `logits_to_keep` works here and does not at the coded readouts, for a
        # reason worth stating: every row is `prefix_ids + option_ids`, so the
        # prefix is shared and each option's tokens start at the same index in
        # every row. The positions needed run `len(prefix_ids) - 1` through
        # `width - 2` -- a common suffix, not a per-row position -- and keeping
        # `width - len(prefix_ids) + 1` of them makes the slice begin at exactly
        # `len(prefix_ids) - 1`.
        output = self.model(input_ids=input_ids, attention_mask=attention,
                            output_hidden_states=bool(self.feature_layers),
                            logits_to_keep=width - len(prefix_ids) + 1)
        logprobs = torch.log_softmax(output.logits.float(), dim=-1)
        scores = [
            OptionScore(index=index, text=text, tokens=len(ids),
                        # Rebased, not re-derived: the element at absolute
                        # `len(prefix_ids) - 1 + offset` is at slice index
                        # `offset`, which is what a prefix length of 1 computes.
                        total=summed_logprob(logprobs[index], 1, ids))
            for index, (text, ids) in enumerate(zip(options, per_option, strict=True))
        ]
        # Row zero and the last prefix position: the prefix is identical in every row
        # and attention is causal, so that vector is the same in all of them. Which is
        # also why a head can exist for this readout at all -- the state is a property
        # of the question, and the options are what each row adds after it.
        #
        # Through the same `_readout_hidden` the coded path uses, because one rule with
        # two implementations is the fault this repository spent today on. It is not
        # the same *state*: that one follows a prompt naming the options with letters
        # and this one a prompt listing them with none, which is what
        # `HeadMetadata.readout` records and what the engine refuses to mix.
        return scores, rewritten, self._readout_hidden(output, 0, len(prefix_ids) - 1)

    def think(self, state, schema, names, budget):
        """Slow path for the named questions: backbone thinking, then the same readout."""
        with self.lock, torch.inference_mode():
            return self._think(state, schema, tuple(names), budget)

    def propose(self, state, instructions, width=10, max_new_tokens=32, temperature=None,
                seed=None, verified=False):
        """Answers this question could have, for a question that arrived with none.

        Thinking stays switched off, so these are the fast mode's own answers and
        not a reasoning trace: a beam of `width` returns that many completions in
        one decode, and `litjev.construct` turns the distinct ones into the
        candidate set the readout then scores.

        The beams are the fast mode's alone. The slow mode renders its own prompt
        and never receives them, which is what keeps it the policy the
        always-thinking baseline measures.
        """
        if temperature:
            raise ValueError(
                "beam search does not sample, so a temperature here would describe "
                "something that did not happen; use the sglang backend to sample")
        if seed is not None:
            raise ValueError(
                "beam search does not sample, so there is no draw for a seed to fix; "
                "the beam is already the same every time")
        if verified:
            raise ValueError(
                "this backend decodes without a grammar, so it cannot be held to the "
                "proposal schema; use the sglang backend, or pass --no-verified-proposals "
                "and accept that the answers are parsed out of free text")
        with self.lock, torch.inference_mode():
            return self._propose(state, instructions, width, max_new_tokens)

    def solve(self, state, question, budget, answer_tokens=48):
        """Think about a question that has no options, and write the answer out.

        `think` renders its prompt through `question_body`, which writes every
        option into it. When the options were constructed by the fast mode that
        turns the slow mode into a policy answering from the fast mode's
        proposals, which is not the policy the always-thinking column measures.
        A budget that runs out mid-thought is made to answer anyway, the way
        `think` does, so that the two slow columns are the same policy.

        Returns `(text, thinking_tokens, truncated)`.
        """
        with self.lock, torch.inference_mode():
            return self._solve(state, question, budget, answer_tokens)

    def _solve(self, state, question, budget, answer_tokens):
        from litjev.core.prompts import build_open_thinking_messages

        text = self.tokenizer.apply_chat_template(
            build_open_thinking_messages(state, question), tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        written, spent = self._complete(text, budget)
        if "</think>" in written:
            return written, spent, False
        forced, extra = self._complete(text + written + "</think>\nAnswer:", answer_tokens)
        return written + "</think>\nAnswer:" + forced, spent + extra, True

    def _complete(self, text, max_new_tokens):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        generated = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self._pad_id())
        written = generated[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(written, skip_special_tokens=True), int(written.shape[0])

    def _propose(self, state, instructions, width, max_new_tokens):
        from litjev.core.prompts import build_proposal_messages

        if width < 2:
            raise ValueError("a beam narrower than two proposes no choice")
        text = self.tokenizer.apply_chat_template(
            build_proposal_messages(state, instructions), tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **ids, num_beams=width, num_return_sequences=width,
            max_new_tokens=max_new_tokens, do_sample=False,
            early_stopping=True, output_scores=True, return_dict_in_generate=True,
            pad_token_id=self._pad_id())
        prompt_length = ids["input_ids"].shape[1]
        scores = (output.sequences_scores.tolist()
                  if getattr(output, "sequences_scores", None) is not None
                  else [0.0] * len(output.sequences))
        beams = []
        for sequence, score in zip(output.sequences[:, prompt_length:].tolist(), scores):
            beams.append((self.tokenizer.decode(sequence, skip_special_tokens=True),
                          float(score)))
        return beams

    def _compile(self, state, schema):
        return compile_slots(self.tokenizer, state, schema, self.max_input_tokens,
                             code_offset=self.code_offset)

    def _render_unified(self, state, schema, names=None, thinking=False):
        """One render per question, cut two ways. `SGLangScorer._render`'s body.

        Kept as the same sequence of the same calls rather than a second
        rendering: `build_messages`, `question_body`, `shared_prefix` and the two
        thinking cuts all live in `litjev.core`, and the point of this path is to
        read features at the prompt that path serves. A paraphrase here would be
        a second definition of the thing being held equal, and the test that
        matters asserts the two produce identical strings.

        `code_offset` is deliberately not applied. It rotates codes against slots
        to ask what the readout reads, and `compile_slots` is where that happens;
        this render has no slots to rotate against, and the SGLang path it mirrors
        does not rotate either, so honouring it here would make the two differ.
        """
        if self.code_offset:
            raise ValueError(
                "code_offset rotates codes against the compiled slots, which the "
                "unified render does not have; the SGLang path it mirrors ignores "
                "it, so a run setting both would be comparing two readouts")
        codes = candidate_codes(self.tokenizer, max(len(f.choices) for f in schema.values()))
        prompts, labels = [], []
        for name, question in schema.items():
            if names is not None and name not in names:
                continue
            own = codes[: len(question.choices)]
            rendered = self.tokenizer.apply_chat_template(
                build_messages(state, question_body(question, own)),
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            shared = shared_prefix(rendered)
            prompts.append(open_thinking(shared) if thinking
                           else close_thinking(shared) + ANSWER_BOUNDARY)
            labels.append((name, own))
        return prompts, labels

    def _score_unified(self, state, schema):
        """The fast mode on the unified render: one row per question, read at the end.

        Right-padded, and each row is read at its own last real token rather than
        at a common final column. Left padding would put every row's readout in
        one place and is the usual way to batch this, but it also makes the
        position ids depend on the pad run, and this readout is the state a head
        is fitted to -- so the arithmetic that has to be right is the index, not
        the padding.
        """
        if isinstance(state, VisualState):
            raise TypeError("the unified render has no image path; it is text-only")
        prompts, labels = self._render_unified(state, schema, thinking=False)
        rows = [self.tokenizer.encode(text, add_special_tokens=False) for text in prompts]
        longest = max(map(len, rows))
        if longest > self.max_input_tokens:
            raise ValueError(
                f"a unified fast prompt is {longest} tokens against a limit of "
                f"{self.max_input_tokens}")
        device = self.model.get_input_embeddings().weight.device
        pad = self._pad_id()
        ids = torch.tensor([row + [pad] * (longest - len(row)) for row in rows], device=device)
        mask = torch.tensor([[1] * len(row) + [0] * (longest - len(row)) for row in rows],
                            device=device)
        output, readout_logits = self._forward_readout(
            [len(row) - 1 for row in rows],
            input_ids=ids, attention_mask=mask, use_cache=False)
        config = getattr(self.model.config, "text_config", self.model.config)
        fields = []
        for i, ((name, codes), row) in enumerate(zip(labels, rows, strict=True)):
            position = len(row) - 1
            candidates = boundary_candidate_ids(self.tokenizer, ANSWER_BOUNDARY, codes)
            fields.append(RawFieldScores(
                name,
                readout_logits[i, candidates].float().cpu().numpy(),
                len(row),
                {
                    # Named so a records file says which of this backend's two
                    # renders produced it. `tab:features` was measured under the
                    # other one and nothing in its records distinguishes them.
                    "method": "one_forward_unified_prompt",
                    "system": "unified",
                    "slot_format": SLOT_FORMAT,
                    "boundary": ANSWER_BOUNDARY.strip(),
                    "codes": list(codes),
                    "module": "lm_head",
                    "batch_index": i,
                    "selected_logit_index": position,
                    "prompt_token_count": len(row),
                    "last_decoder_layer_index": config.num_hidden_layers - 1,
                    "decoder_layer_count": config.num_hidden_layers,
                    "candidate_labels": list(schema[name].choices),
                    "candidate_codes": list(codes),
                    "candidate_token_ids": candidates,
                    "feature_layers": list(self.feature_layers),
                    "observation_modality": "text",
                },
                hidden=self._readout_hidden(output, i, position),
            ))
        return tuple(fields)

    def _pad_id(self):
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError("Tokenizer requires a padding or EOS token")
        return pad

    def _forward_readout(self, positions, **kwargs):
        """Hidden states, and logits only at the positions a readout reads.

        The causal LM's forward materialises `batch x width x vocab` logits and
        every readout below takes one position per row. On the 27B at batch 5
        over a 5323-token sequence that is 12.3 GiB allocated to use 2.4 MiB of
        it, and it is the allocation that killed two collection shards: the
        exception asked for 12.31 GiB with 10.68 free. The base transformer
        returns the same hidden states -- `XForCausalLM.forward` passes its
        `hidden_states` through unchanged -- so the head is applied to the
        gathered rows instead, turning (B,L,H)x(H,V) into (B,H)x(H,V).

        Right padding is why `logits_to_keep=1` will not do: each row's readout
        is at its own last real token, not at the last column.

        The cached path in `_score` is left on the causal LM. Its suffixes are
        short and share a prefix, so its logits are small, and it reads
        `past_key_values` back.
        """
        inner = getattr(self.model, "model", None)
        head = getattr(self.model, "lm_head", None)
        if inner is None or head is None:
            raise RuntimeError(
                f"{type(self.model).__name__} has no .model/.lm_head pair, so the readout "
                "cannot skip the full-sequence logits. Falling back silently would restore "
                "a 12 GiB allocation on the 27B without saying so.")
        self._refuse_post_head_transforms()
        base = inner(
            **kwargs,
            output_hidden_states=bool(self.feature_layers),
        )
        states = base.last_hidden_state
        rows = torch.arange(len(positions), device=states.device)
        index = torch.as_tensor(list(positions), device=states.device)
        return base, head(states[rows, index])

    # Families that transform the logits *after* `lm_head`, which is the one
    # assumption `_forward_readout` makes and cannot check from the tensors.
    # Qwen -- every model this paper reports -- applies nothing, so the readout is
    # exact there. Gemma-2/3 and VaultGemma softcap and Cohere scales, and both
    # would pass unnoticed: softcapping is monotone, so the answer argmax is
    # unchanged and only the *distribution* moves, which is precisely what the
    # decision head reads. A silent drop would therefore shift every
    # `distribution_stats` feature while leaving accuracy identical.
    POST_HEAD_TRANSFORMS = ("final_logit_softcapping", "logit_scale", "logits_scaling")

    def _refuse_post_head_transforms(self):
        # `get_text_config()` is what the forward itself reads -- gemma-4's is
        # `config.get_text_config().final_logit_softcapping` -- and on a
        # multimodal config it is not always reachable as a `text_config`
        # attribute, so the accessor is preferred and the attribute is the
        # fallback for configs too old to have it.
        root = self.model.config
        accessor = getattr(root, "get_text_config", None)
        config = accessor() if callable(accessor) else getattr(root, "text_config", root)
        declared = [name for name in self.POST_HEAD_TRANSFORMS
                    if getattr(config, name, None) is not None
                    # A multimodal config that nests nothing returns the root, so
                    # check the root too rather than trusting one of the two.
                    or getattr(root, name, None) is not None]
        if declared:
            raise RuntimeError(
                f"{type(self.model).__name__} declares {declared}, which its forward applies "
                "after lm_head. Applying the head alone would change every distribution "
                "feature the decision head reads while leaving the answer argmax intact, so "
                "it would not show up as an accuracy change. Score this backbone through the "
                "causal LM, or reproduce the transform here.")

    def _readout_hidden(self, output, row, position):
        if not self.feature_layers:
            return None
        states = output.hidden_states
        if states is None:
            raise RuntimeError("Model did not return hidden states")
        return (
            torch.stack([states[layer][row, position] for layer in self.feature_layers])
            .float()
            .cpu()
            .numpy()
        )

    def _prepare(self, state, schema):
        device = self.model.get_input_embeddings().weight.device
        if isinstance(state, VisualState):
            if self.processor is None or self.model.config.model_type != "qwen3_5":
                raise ValueError("Image decisions require a Qwen qwen3_5 model and processor")
            validate_image(state.image)
            messages = build_decision_messages(state.text, schema)
            messages[-1]["content"] = [
                {"type": "text", "text": state.text},
                {"type": "image", "image": state.image.convert("RGB")},
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
            if "pixel_values" not in inputs or "mm_token_type_ids" not in inputs:
                raise ValueError("Processor must return pixels and multimodal token types")
            compiled = compile_prefix_slots(
                self.tokenizer,
                schema,
                inputs["input_ids"][0].tolist(),
                max_input_tokens=self.max_input_tokens,
                code_offset=self.code_offset,
            )
            return compiled, {key: value.to(device) for key, value in inputs.items()}
        compiled = self._compile(state, schema)
        prefix = compiled.input_ids[0][: compiled.prefix_length]
        prefix_ids = torch.tensor([prefix], device=device)
        return compiled, {"input_ids": prefix_ids, "attention_mask": torch.ones_like(prefix_ids)}

    def _score(self, state, schema):
        compiled, inputs = self._prepare(state, schema)
        device = inputs["input_ids"].device
        prefix_length = compiled.prefix_length
        base = self.model(**inputs, use_cache=True, logits_to_keep=1)
        cache = base.past_key_values
        reorder = getattr(cache, "reorder_cache", None)
        if reorder is None:
            raise RuntimeError("Model cache does not support branch replication")
        # Supports Qwen hybrid attention: both KV and convolution/recurrent states.
        reorder(torch.zeros(len(schema), dtype=torch.long, device=device))
        suffixes = compiled.slot_ids
        width = max(map(len, suffixes))
        pad = self._pad_id()
        ids = torch.tensor([row + [pad] * (width - len(row)) for row in suffixes], device=device)
        lengths = torch.tensor(list(map(len, suffixes)), device=device)
        suffix_mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
        mask = torch.cat(
            [
                torch.ones((len(schema), prefix_length), device=device, dtype=torch.long),
                suffix_mask.long(),
            ],
            dim=1,
        )
        positions = torch.arange(prefix_length, prefix_length + width, device=device)
        positions = positions[None, :].expand(len(schema), -1)
        if isinstance(state, VisualState):
            # Image patches consume sequence slots but have 3-D rotary coordinates.
            # Continue after the image prefix's M-RoPE extent, not its token count.
            delta = self.model.model.rope_deltas
            if delta is None or delta.shape[0] != 1:
                raise RuntimeError("Missing single-image-prefix M-RoPE state")
            positions = (positions + delta.to(device))[None, :, :].expand(3, -1, -1)
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=bool(self.feature_layers),
        )
        config = getattr(self.model.config, "text_config", self.model.config)
        return tuple(
            RawFieldScores(
                name,
                output.logits[i, len(suffixes[i]) - 1, compiled.candidates[i]]
                .float()
                .cpu()
                .numpy(),
                prefix_length + sum(map(len, suffixes)),
                {
                    "method": "two_forward_cached_branches",
                    "system": "one",
                    "slot_format": SLOT_FORMAT,
                    "module": "lm_head",
                    "last_decoder_layer_index": config.num_hidden_layers - 1,
                    "decoder_layer_count": config.num_hidden_layers,
                    "batch_index": i,
                    "selected_logit_index": len(suffixes[i]) - 1,
                    "prefix_token_count": compiled.prefix_length,
                    "slot_text": compiled.slot_texts[i],
                    "slot_token_ids": compiled.slot_ids[i],
                    "absolute_position": compiled.positions[i],
                    "candidate_labels": list(schema[name].choices),
                    "candidate_codes": compiled.candidate_codes[i],
                    "candidate_token_ids": compiled.candidates[i],
                    "feature_layers": list(self.feature_layers),
                    "observation_modality": "image" if isinstance(state, VisualState) else "text",
                    "image_grid_thw": inputs["image_grid_thw"].tolist()
                    if "image_grid_thw" in inputs
                    else None,
                    "readout_rope_positions": positions[:, i, len(suffixes[i]) - 1].tolist()
                    if positions.ndim == 3
                    else None,
                },
                hidden=self._readout_hidden(output, i, len(suffixes[i]) - 1),
            )
            for i, name in enumerate(schema.names)
        )

    def _think(self, state, schema, names, budget):
        if isinstance(state, VisualState):
            raise TypeError("The slow path does not support image states yet")
        if budget <= 0:
            raise ValueError("Thinking budget must be positive")
        unknown = [name for name in names if name not in schema]
        if unknown or not names:
            raise ValueError("Slow path names must be non-empty schema question IDs")
        compiled = self._compile(state, schema)
        device = self.model.get_input_embeddings().weight.device

        def encode(text):
            return self.tokenizer.encode(text, add_special_tokens=False)

        open_text, close_text = self.think_tokens
        close_ids = encode(close_text)
        tail_ids = encode("\n" + close_text + "\n" + ANSWER_BOUNDARY)
        if not encode(open_text) or not close_ids:
            raise ValueError("Tokenizer cannot encode the thinking delimiters")
        index = {name: i for i, name in enumerate(schema.names)}
        prompts = []
        if self.unified_prompt:
            # The same cut the fast pass took, with the block left open. Both
            # modes have to come from one render or the flag buys nothing: a fast
            # pass under the unified prompt with a slow pass under
            # `build_thinking_messages` would read the features at one prompt and
            # the label at another, which is the fault this exists to remove.
            #
            # Mapped by name, not by position. `_render_unified` walks the schema
            # and `names` is the caller's order, so zipping the two lists would
            # pair a prompt with another question's record wherever a caller
            # escalated a subset out of schema order -- and nothing would fail.
            rendered, labels = self._render_unified(state, schema, names=names, thinking=True)
            by_name = {name: text for (name, _codes), text in zip(labels, rendered, strict=True)}
            missing = [name for name in names if name not in by_name]
            if missing:
                raise ValueError(f"the unified render produced no prompt for {missing}")
            prompts = [encode(by_name[name]) for name in names]
        else:
            for name in names:
                i = index[name]
                # The slow prompt is rendered with thinking enabled, so it carries no
                # empty <think></think> block; Qwen's template then ends the prompt
                # with "<think>\n".
                text = self.tokenizer.apply_chat_template(
                    build_thinking_messages(
                        state, question_body(schema[name], compiled.candidate_codes[i])
                    ),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                if not text.rstrip().endswith(open_text):
                    text += open_text + "\n"
                prompts.append(encode(text))
        if max(map(len, prompts)) + budget + len(tail_ids) > self.max_input_tokens:
            raise ValueError("Thinking budget exceeds the input token limit")
        pad = self._pad_id()
        width = max(map(len, prompts))
        ids = torch.tensor([[pad] * (width - len(row)) + row for row in prompts], device=device)
        mask = torch.tensor(
            [[0] * (width - len(row)) + [1] * len(row) for row in prompts], device=device
        )
        generated = self.model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=budget,
            do_sample=False,
            pad_token_id=pad,
            stopping_criteria=StoppingCriteriaList([SuffixStop(close_ids, width)]),
        )
        eos = self.tokenizer.eos_token_id
        thoughts, sequences = [], []
        for row, prompt in zip(generated[:, width:].tolist(), prompts, strict=True):
            while row and row[-1] in {pad, eos}:
                row.pop()
            if len(row) >= len(close_ids) and row[-len(close_ids) :] == close_ids:
                row = row[: -len(close_ids)]
            thoughts.append(row)
            sequences.append(prompt + row + tail_ids)
        width = max(map(len, sequences))
        ids = torch.tensor([row + [pad] * (width - len(row)) for row in sequences], device=device)
        mask = torch.tensor(
            [[1] * len(row) + [0] * (width - len(row)) for row in sequences], device=device
        )
        output, readout_logits = self._forward_readout(
            [len(row) - 1 for row in sequences],
            input_ids=ids, attention_mask=mask, use_cache=False)
        decode = getattr(self.tokenizer, "decode", None)
        results = []
        for j, name in enumerate(names):
            i = index[name]
            position = len(sequences[j]) - 1
            # Under the unified render there are no compiled slots to take the
            # codes from, so they come from the same two calls the fast pass made.
            # They should agree with `compiled`'s -- both start at
            # `candidate_codes`, and `code_offset`, the one thing that would move
            # them apart, is refused by `_render_unified`. Recomputed rather than
            # assumed equal, because a silent disagreement here reads the slow
            # mode's logits at another question's codes.
            own = (list(candidate_codes(self.tokenizer,
                                        max(len(f.choices) for f in schema.values()))
                        )[: len(schema[name].choices)]
                   if self.unified_prompt else compiled.candidate_codes[i])
            candidates = (boundary_candidate_ids(self.tokenizer, ANSWER_BOUNDARY, own)
                          if self.unified_prompt else compiled.candidates[i])
            results.append(
                RawFieldScores(
                    name,
                    readout_logits[j, candidates].float().cpu().numpy(),
                    len(sequences[j]),
                    {
                        "method": "slow_thinking_full_input",
                        "slow_prompt_format": ("unified_one_render_v1" if self.unified_prompt
                                               else "user_turn_question_thinking_v1"),
                        "system": "unified" if self.unified_prompt else "two",
                        "slot_format": SLOT_FORMAT,
                        "module": "lm_head",
                        "selected_logit_index": position,
                        "candidate_labels": list(schema[name].choices),
                        "candidate_codes": own,
                        "candidate_token_ids": candidates,
                        "thinking_tokens": len(thoughts[j]),
                        "thinking_text": decode(thoughts[j]) if decode else None,
                        "budget": budget,
                        "feature_layers": list(self.feature_layers),
                    },
                    hidden=self._readout_hidden(output, j, position),
                    generated_tokens=len(thoughts[j]),
                )
            )
        return tuple(results)
