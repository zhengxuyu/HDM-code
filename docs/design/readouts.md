# How the fast mode reads its answer

Two readouts exist. They differ in one thing: whether the candidate answers go
into the model's context, or only into the prompt that precedes it. Everything
else — the cost, the positional bias, the fact that one needs letters and the
other does not — follows from that.

Written by `exps` on 2026-10-01 from the implementation, not from the design
notes: `TransformersScorer.score_options`, `litjev.prompting` (the prompt) and
`SGLangScorer.score_options` (the SGLang port) -- named by symbol, because line
numbers move. Figures
are measured, with their job ids.

---

## The coded readout: one forward, k lookups in the vocabulary

The options are listed **with letters**, the prompt ends at `Answer:`, and one
forward pass gives the distribution over the next token. The readout reads the
`k` entries belonging to the `k` code tokens and takes the argmax among them.

```
prompt (options listed WITH letters)                 logits at the last position
┌──────────────────────────────────────┐            ┌────────────────────────┐
│ Question: ...                        │            │  vocabulary ~150k      │
│ A. Oxygen                            │  one       │   ...                  │
│ B. Osmium                 ───────────────────────▶│   "A" → -2.1   ◀── read│
│ ...                                  │  forward   │   "B" → -0.4   ◀── read│
│ J. ...                               │            │   ...                  │
│ Answer:                              │            │   "J" → -5.3   ◀── read│
└──────────────────────────────────────┘            └────────────────────────┘
                                                      argmax over those k
```

**Cost: one forward pass.** 0.0789 s per question, 27B on SGLang over all 1500 (28284).

This is the readout every number in the paper's tables came from.

---

## The content readout: k rows in one batched prefill

The options are listed **with no letters at all**, one per line, and the prompt
still ends at `Answer:`. Then each option's own text is appended to that same
prefix, teacher-forced, and the readout sums the log-probabilities of **the
option's own tokens**.

```
prefix = [chat template] + "Question: ..." + options, one per line, NO letters
                         + "Answer:"          <-- no trailing space, see below

         ├─────────── shared prefix, ~900 tokens ───────────┤├ the option's own tokens ┤
row 0:   prefix_ids ..................................... " Oxy" "gen"            pad pad
row 1:   prefix_ids ..................................... " Os" "mi" "um"             pad
row 2:   prefix_ids ..................................... " a" " gas" " that" " ..."
  ...
row 9:   prefix_ids ..................................... " ..." ...
         └──────── right-padded; left-padding would move the prefix, and the
                   prefix is where every option's first token is predicted from

                                   ↓
                  one model(input_ids) → logits → log_softmax
                                   ↓
     per row, take only the positions belonging to that option and sum them:
          summed_logprob(logprobs[row], len(prefix_ids), option_ids)
```

Each row yields:

| quantity | definition |
| --- | --- |
| `sum` | Σ logprob over the option's own tokens |
| `tokens` | how many tokens that was |
| `mean` | `sum / tokens` |
| `prior` | the same option's `sum` after the question is replaced by a content-free instruction — a **second** batched prefill |
| `pmi` | `sum - prior` |

So `mean` and `pmi` are not independent measurements: the independent quantities
per option are **`sum`, `prior`, `tokens`**, and the position index.

### Why it cannot read one logit vector instead

Because an option is not one token. Measured over the 14 336 options of the
stride-8 1500 (28037):

| | tokens per option |
| --- | --- |
| median | **8** |
| mean | 10.7 |
| 25% / 75% / 90% | 5 / 14 / 21 |
| longest | 111 |
| **exactly one token** | **2.13%** |

The logit vector at the prompt's last position gives the distribution over the
**first** continuation token only. The second token's conditional log-probability
depends on the first already being in the context, and that is not recoverable
from the prefix's last position. So each option's sequence has to enter the
context, which is what the `k` rows are for.

A readout that reads `k` entries from one vector is therefore only possible when
the options are single tokens — which is what the coded readout achieves by
making the answer a letter.

### Why the prompt must not end in a space

`unlabelled_suffix` ends at the colon with **no trailing space**, and the
separating space is prepended to the option instead. Under BPE a trailing space
is its own token while the same space binds to the following word — `" Oxygen"`
is one token — so the joint encoding of prompt and option would not begin with
the prompt's own tokens and the option could not be separated from it. Measured
on gpt2: with the trailing space none of seven ordinary options is separable,
without it all seven are. This was `65d8a84`'s red flag: with the real
Qwen3.5-4B tokenizer the merge made `option_token_ids` raise on 7 of 8 typical
options, and the stub tokenizer in the tests was character-level and could not
have shown it.

---

## The same readout on two backends

Both do `k` sequences through the context. They differ in how those `k` reach the
engine.

```
transformers      k rows padded into one tensor  →  ONE model() call
                  1.0697 s/question (28037)

SGLang            k separate requests, the prefix cache avoiding recomputation
                  1.1945 s/question (28284), ~125 ms per option
```

9.56 options × 125 ms ≈ 1.19 s, which is what was measured. **The prefix cache
saves recomputation, not per-request overhead.**

All figures below are over all 1500 questions. An earlier version of this file
quoted 0.136 / 1.690 / 12.5×, which were the first 254 rows: both absolute times
fall as a run warms up and `coded` falls faster, so the ratio climbs — 12.3× over
the first 254, 16.5× over the last 254, 15.1× pooled. paper-writer caught it from
the cluster; it is the same early-sample error recorded against 28031 in EXP.md.

| | coded | content | ratio, mean | ratio, median |
| --- | --- | --- | --- | --- |
| transformers (28037) | 0.4263 s | 1.0697 s | **2.5×** | 4.5× |
| SGLang (28284) | **0.0789 s** | 1.1945 s | **15.1×** | 18.1× |

**The ratio grows because the denominator shrinks, not because content slows
down.** SGLang makes `coded` **5.4× faster** (0.4263 → 0.0789) and `content`
**1.12× slower** (1.0697 → 1.1945). So "the content readout is slow" is the wrong
reading: SGLang speeds up the single-forward readout enormously and the
k-sequence one barely at all.

`load_proposer` hard-codes `batch_questions=False`, for the misattribution risk
rather than as a variable held equal to 28037 — and it is not held equal in any
case, since transformers puts all k options in one `model()` call while SGLang
sends k requests, so the two runs differ in that as well as in backend. **The
batched SGLang path has never been run**, so 2.5× is not a target SGLang has been
shown to reach, and whether the gap is removable is untested. `74a4b74` makes
`verify` compare the option scores both ways against one reference, which is the
check that would settle it, and it needs a GPU.

### The backends agree on the decision, not on the value

28279/28280, 27B: cosine min 0.99952, relative error max 0.03102, and for the
content readout `same_choice: true` on all three questions with
`|sum difference|` max 0.12567. Sums that had matched would have meant the
comparison was not running — relative error maxes at ~3% on a single activation
and an option score sums several positions. Only the choice can be required to
survive. `hidden_states_ndim: 3` was measured in the same run.

---

## What the content readout buys

| | coded | content (`sum`) |
| --- | --- | --- |
| 27B accuracy, 1500 questions | 0.6107 | **0.6460** — paired 151:98, **p = 0.0009** |
| 4B accuracy | 0.4773 | 0.4947 (p = 0.16; `mean` 0.5007 at p = 0.0545) |
| 27B positional bias | 0.385 | **0.172** |
| 4B positional bias | 0.312 | **0.185** |
| favoured slot | **moves with the model**: s1 at 4B (letter B), s0 at 27B | s0 on both |

Bias is the off-target pick rate `P(pick slot p | gold ≠ p)` over ten-option
questions, summarised as its total variation from uniform — the accuracy-free
statistic, because raising accuracy pushes `chosen` toward `gold` by itself.

The favoured slot moving between backbones is why the coded readout cannot be
described in one sentence in a paper: "a prior on the second listed option" is
true of the 4B and false of the 27B.

### `sum` against `mean`

No basis to choose on accuracy: 26:35 at 4B (p = 0.31) and 53:41 at 27B
(p = 0.26), opposite directions, agreeing on the same option 92.6% and 90.3% of
the time. Three other axes agree on both backbones and all favour `sum`:

| | 4B | 27B |
| --- | --- | --- |
| AUROC of the top-two margin for its own correctness | **0.758** vs 0.698 | **0.833** vs 0.764 |
| within-question correlation with token count | **+0.079** vs +0.419 | **+0.110** vs +0.480 |
| accuracy where gold is the shortest option | **0.571** vs 0.500 | **0.701** vs 0.591 |

The length axis is the one with a transfer argument behind it. Gold is longer
than average in every benchmark measured, but by amounts that differ and with
subtasks where it reverses: MMLU-Pro +0.247 sd, GPQA-Diamond +0.124, AGIEval
+0.086, and agieval/sat-en −0.100, mmlu-pro/computer science −0.064. A readout
that barely sees length transfers; one that leans on it gains where gold is long
and pays where it is short.

### `pmi` is a negative result, not an absence of bias

Worst accuracy on both backbones by ~13 points (0.3680 and 0.5127), least biased
on both (0.143 and 0.098), and it **avoids** slot 0 where everything else prefers
it (0.033 and 0.025). Its low bias is a reversal.

It is also not removing a length effect: correlation with token count is `prior`
−0.048, `sum` +0.030, `pmi` +0.079 — all ~0. What it divides out is how commonly
the option is said irrespective of the question, and on this data that is
informative: 103 questions where `pmi` is right and `sum` wrong, against **293**
the other way.

---

## The cost objection, and its denominator

Content costs 1.55 s/question more than coded on SGLang today. A slow-thinking
step is ~34 s (median `thinking_tokens` 1024, capped on 1354 of 1500 in
records-4b-27063). So the extra readout is **4.6% of one thinking step**, and a
system that routes:

| fraction sent to thinking | coded | content | ratio |
| --- | --- | --- | --- |
| 5% | 1.84 s | 3.40 s | 1.84× |
| 10% | 3.55 s | 5.10 s | 1.44× |
| 20% | 6.96 s | 8.52 s | 1.22× |
| 50% | 17.20 s | 18.76 s | 1.09× |

15.1× is a ratio between readouts, not between systems -- and most of it is
SGLang's speed-up of `coded`, not a cost of `content`.

---

## The option nobody has measured: score the first token only

Read `k` entries from the one logit vector, as the coded readout does, but read
the **options' own first tokens** rather than letters.

```
prefix (no letters) → one forward → logits at last position
                    → read the first token of each option → argmax
```

Cost equals coded. Output is content, not an index, so it satisfies the
constraint that the no-thinking answer must never be `A/B/C/D` or `1/2/3/4`.
Whether it works depends on how often the `k` options' first tokens are
distinct — and that is computable with no GPU from the datasets alone. **Not yet
measured.**

---

## Why letters cannot simply be added back

On an enumerated benchmark like MMLU-Pro, numbering the options is harmless: the
options exist and the letter is a pointer to one of them.

On a free-form benchmark like MATH-500 the candidates are proposed by the model
itself. Labelling them and reading out the label makes the answer an index
again, which is the failure `fa0a6c7` fixed: candidates were keyed by position,
so the engine returned `"1"` where the answer was `"5"`, and
`open_three_policies` mapped it back to the text itself — so the one caller
anybody read recorded correct answers while the engine returned the index to
everybody else.

So "add numbering automatically" is not a unified scheme. It works where options
are given and reintroduces the index where they are constructed.
