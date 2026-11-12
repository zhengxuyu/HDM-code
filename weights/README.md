# Self-routing blocks, fitted on MMLU-Pro

Also on the Hub: <https://huggingface.co/fredreick/HDM-blocks>

Two blocks, one per backbone. Each reads the hidden state at the readout position
of the fast pass together with eight statistics of the fast answer, and returns a
distribution over the four joint outcomes of the two modes. The value of thinking
and the fast mode's confidence are the two marginals of that distribution.

Both were fitted with the supervised objective on MMLU-Pro and then used
everywhere else unchanged: nothing in this repository retrains them per benchmark.

| | `head-4b-mmlu-pro` | `head-27b-mmlu-pro` |
|---|---|---|
| backbone | Qwen3.5-4B | Qwen3.8-27B |
| readout | `coded` | `coded` |
| layer | `-1`, forced | `-1`, forced |
| projection | none (raw hidden state) | none (raw hidden state) |
| width | 2560 + 8 | 5120 + 8 |
| file | 2.6 MB | 5.3 MB |

## What they were fitted on

MMLU-Pro, 1,154 questions over eleven subject categories, with chemistry,
economics and psychology held out: 346 questions the block never saw, split at
the level of the category rather than the question. Supervised objective, seed 0.

| on the held-out categories | 4B | 27B |
|---|---|---|
| fast mode accuracy | 0.543 | 0.650 |
| slow mode accuracy | 0.853 | 0.887 |
| thinking helps | 0.335 | 0.272 |
| thinking hurts | 0.026 | 0.035 |
| AUROC, value of thinking against where thinking helps | 0.804 | 0.879 |
| AUROC, fast answer correct | 0.797 | 0.868 |

The last two rows are what the block is for. The 4B figures are worth reading
beside the baselines in the report files beside these weights: on that backbone a
probe over the eight statistics alone scores 0.815 on the first of them, slightly
above the block, and the fast mode's own top probability scores 0.841 on the
second. We read the raw last layer anyway, for a reason that is not accuracy:
every transformer has a last layer, and a projection fitted on one backbone's
features is not portable to another.

`head-*-report.json` beside each file carries the full fit, including those
baselines and the per-layer selection the forced layer overrode.

## Loading

```python
from litjev.core.block import DecisionHead

block = DecisionHead.load("weights/head-27b-mmlu-pro.safetensors")
```

The serving path refuses a pairing the block was not fitted for: `litjev.serving`
compares the model id, the revision, the readout and the hidden size before the
block is attached, so a block served against the wrong backbone raises rather
than reading noise.

## A block is fitted to its features, not to a task

Changing the backbone, the layer, the readout or the numeric precision changes
the distribution these were fitted on, and the calibration goes with it. A block
is a few hundred kilobytes and fits in minutes, so refitting is the ordinary
move, not an exceptional one.
